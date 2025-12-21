#define _USE_MATH_DEFINES

#include <array>
#include <cstdint>
#include <climits>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <random>

#include <veekay/veekay.hpp>

#include <imgui.h>
#include <vulkan/vulkan_core.h>

namespace {

constexpr float camera_fov = 70.0f;
constexpr float camera_near_plane = 0.01f;
constexpr float camera_far_plane = 100.0f;

struct Matrix {
	float m[4][4];
};

struct Vector {
	float x, y, z;
};

struct Vertex {
	Vector position;
	Vector color;
};

struct ShaderConstants {
	Matrix projection;
	Matrix transform;
	// Vector color;
};

struct VulkanBuffer {
	VkBuffer buffer;
	VkDeviceMemory memory;
};

struct Mesh {
	VulkanBuffer vertex_buffer;
	VulkanBuffer index_buffer;
	uint32_t index_count;
};

struct SceneObject {
	int id;

	SceneObject *parent;

	Vector position = {0, 0, 5};
	Vector scale = {1, 1, 1};
	float rotation = 0;
	float rotation_speed = 5;
	bool spin = true;
	Vector rotation_axis = {0.5f, 1, 0};

	Mesh mesh;
};

VkShaderModule vertex_shader_module;
VkShaderModule fragment_shader_module;
VkPipelineLayout pipeline_layout;
VkPipeline pipeline;

std::vector<SceneObject> objects;

double time_prev = 0;
float dt;


Matrix identity() {
	Matrix result{};

	result.m[0][0] = 1.0f;
	result.m[1][1] = 1.0f;
	result.m[2][2] = 1.0f;
	result.m[3][3] = 1.0f;
	
	return result;
}

Matrix projection(float fov, float aspect_ratio, float near, float far) {
	Matrix result{};

	const float radians = fov * M_PI / 180.0f;
	const float cot = 1.0f / tanf(radians / 2.0f);

	result.m[0][0] = cot / aspect_ratio;
	result.m[1][1] = cot;
	result.m[2][3] = 1.0f;

	result.m[2][2] = far / (far - near);
	result.m[3][2] = (-near * far) / (far - near);

	return result;
}

Matrix translation(const Vector vector) {
	Matrix result = identity();

	result.m[3][0] = vector.x;
	result.m[3][1] = vector.y;
	result.m[3][2] = vector.z;

	return result;
}

Matrix rotation(Vector axis, float angle) {
	Matrix result{};

	float length = sqrtf(axis.x * axis.x + axis.y * axis.y + axis.z * axis.z);

	axis.x /= length;
	axis.y /= length;
	axis.z /= length;

	const float sina = sinf(angle);
	const float cosa = cosf(angle);
	const float cosv = 1.0f - cosa;

	result.m[0][0] = (axis.x * axis.x * cosv) + cosa;
	result.m[0][1] = (axis.x * axis.y * cosv) + (axis.z * sina);
	result.m[0][2] = (axis.x * axis.z * cosv) - (axis.y * sina);

	result.m[1][0] = (axis.y * axis.x * cosv) - (axis.z * sina);
	result.m[1][1] = (axis.y * axis.y * cosv) + cosa;
	result.m[1][2] = (axis.y * axis.z * cosv) + (axis.x * sina);

	result.m[2][0] = (axis.z * axis.x * cosv) + (axis.y * sina);
	result.m[2][1] = (axis.z * axis.y * cosv) - (axis.x * sina);
	result.m[2][2] = (axis.z * axis.z * cosv) + cosa;

	result.m[3][3] = 1.0f;

	return result;
}

Matrix multiply(const Matrix& a, const Matrix& b) {
	Matrix result{};

	for (int j = 0; j < 4; j++) {
		for (int i = 0; i < 4; i++) {
			for (int k = 0; k < 4; k++) {
				result.m[j][i] += a.m[j][k] * b.m[k][i];
			}
		}
	}

	return result;
}

Matrix scaling(const Vector factor) {
	Matrix result = identity();

	result.m[0][0] = factor.x;
	result.m[1][1] = factor.y;
	result.m[2][2] = factor.z;

	return result;
}

std::vector<Vector> getConeVertices(const int vert_n) {
	std::vector<Vector> result{};

	for (float t = 0; t + 0.0001f < 2 * M_PI; t += (2 * M_PI) / (vert_n - 2)) {
		result.push_back({std::sin(t), 0, std::cos(t)});
	}

	result.push_back({0, -1, 0});

	result.push_back({0, 0, 0});

	return result;
}

std::vector<uint32_t> getConeIndices(const int vert_n) {
	std::vector<uint32_t> indices{};

	for (int i = 0; i < vert_n - 2; ++i) {

		// Pike
		indices.push_back(i);
		indices.push_back((i + 1) % (vert_n - 2));
		indices.push_back(vert_n - 2);

		// Base
		indices.push_back(vert_n - 1);
		indices.push_back((i + 1) % (vert_n - 2));
		indices.push_back(i);
	}

	return indices;
}

// NOTE: Loads shader byte code from file
// NOTE: Your shaders are compiled via CMake with this code too, look it up
VkShaderModule loadShaderModule(const char* path) {
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	size_t size = file.tellg();
	std::vector<uint32_t> buffer(size / sizeof(uint32_t));
	file.seekg(0);
	file.read(reinterpret_cast<char*>(buffer.data()), size);
	file.close();

	VkShaderModuleCreateInfo info{
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = size,
		.pCode = buffer.data(),
	};

	VkShaderModule result;
	if (vkCreateShaderModule(veekay::app.vk_device, &
	                         info, nullptr, &result) != VK_SUCCESS) {
		return nullptr;
	}

	return result;
}

VulkanBuffer createBuffer(size_t size, void *data, VkBufferUsageFlags usage) {
	VkDevice& device = veekay::app.vk_device;
	VkPhysicalDevice& physical_device = veekay::app.vk_physical_device;
	
	VulkanBuffer result{};

	{
		// NOTE: We create a buffer of specific usage with specified size
		VkBufferCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = size,
			.usage = usage,
			.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		};

		if (vkCreateBuffer(device, &info, nullptr, &result.buffer) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan buffer\n";
			return {};
		}
	}

	// NOTE: Creating a buffer does not allocate memory,
	//       only a buffer **object** was created.
	//       So, we allocate memory for the buffer

	{
		// NOTE: Ask buffer about its memory requirements
		VkMemoryRequirements requirements;
		vkGetBufferMemoryRequirements(device, result.buffer, &requirements);

		// NOTE: Ask GPU about types of memory it supports
		VkPhysicalDeviceMemoryProperties properties;
		vkGetPhysicalDeviceMemoryProperties(physical_device, &properties);

		// NOTE: We want type of memory which is visible to both CPU and GPU
		// NOTE: HOST is CPU, DEVICE is GPU; we are interested in "CPU" visible memory
		// NOTE: COHERENT means that CPU cache will be invalidated upon mapping memory region
		const VkMemoryPropertyFlags flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
		                                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

		// NOTE: Linear search through types of memory until
		//       one type matches the requirements, thats the index of memory type
		uint32_t index = UINT_MAX;
		for (uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
			const VkMemoryType& type = properties.memoryTypes[i];

			if ((requirements.memoryTypeBits & (1 << i)) &&
			    (type.propertyFlags & flags) == flags) {
				index = i;
				break;
			}
		}

		if (index == UINT_MAX) {
			std::cerr << "Failed to find required memory type to allocate Vulkan buffer\n";
			return {};
		}

		// NOTE: Allocate required memory amount in appropriate memory type
		VkMemoryAllocateInfo info{
			.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
			.allocationSize = requirements.size,
			.memoryTypeIndex = index,
		};

		if (vkAllocateMemory(device, &info, nullptr, &result.memory) != VK_SUCCESS) {
			std::cerr << "Failed to allocate Vulkan buffer memory\n";
			return {};
		}

		// NOTE: Link allocated memory with a buffer
		if (vkBindBufferMemory(device, result.buffer, result.memory, 0) != VK_SUCCESS) {
			std::cerr << "Failed to bind Vulkan  buffer memory\n";
			return {};
		}

		// NOTE: Get pointer to allocated memory
		void* device_data;
		vkMapMemory(device, result.memory, 0, requirements.size, 0, &device_data);

		memcpy(device_data, data, size);

		vkUnmapMemory(device, result.memory);
	}

	return result;
}

void destroyBuffer(const VulkanBuffer& buffer) {
	VkDevice& device = veekay::app.vk_device;

	vkFreeMemory(device, buffer.memory, nullptr);
	vkDestroyBuffer(device, buffer.buffer, nullptr);
}

void initialize() {
	VkDevice& device = veekay::app.vk_device;
	VkPhysicalDevice& physical_device = veekay::app.vk_physical_device;

	{ // NOTE: Build graphics pipeline
		vertex_shader_module = loadShaderModule("./shaders/shader.vert.spv");
		if (!vertex_shader_module) {
			std::cerr << "Failed to load Vulkan vertex shader from file\n";
			veekay::app.running = false;
			return;
		}

		fragment_shader_module = loadShaderModule("./shaders/shader.frag.spv");
		if (!fragment_shader_module) {
			std::cerr << "Failed to load Vulkan fragment shader from file\n";
			veekay::app.running = false;
			return;
		}

		VkPipelineShaderStageCreateInfo stage_infos[2];

		// NOTE: Vertex shader stage
		stage_infos[0] = VkPipelineShaderStageCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vertex_shader_module,
			.pName = "main",
		};

		// NOTE: Fragment shader stage
		stage_infos[1] = VkPipelineShaderStageCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = fragment_shader_module,
			.pName = "main",
		};

		// NOTE: How many bytes does a vertex take?
		VkVertexInputBindingDescription buffer_binding{
			.binding = 0,
			.stride = sizeof(Vertex),
			.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
		};

		// NOTE: Declare vertex attributes
		VkVertexInputAttributeDescription attributes[] = {
			{
				.location = 0, // NOTE: First attribute
				.binding = 0, // NOTE: First vertex buffer
				.format = VK_FORMAT_R32G32B32_SFLOAT, // NOTE: 3-component vector of floats
				.offset = offsetof(Vertex, position), // NOTE: Offset of "position" field in a Vertex struct
			},
			{
				.location = 1,
				.binding = 0,
				.format = VK_FORMAT_R32G32B32_SFLOAT,
				.offset = offsetof(Vertex, color),
			}
		};

		// NOTE: Bring 
		VkPipelineVertexInputStateCreateInfo input_state_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
			.vertexBindingDescriptionCount = 1,
			.pVertexBindingDescriptions = &buffer_binding,
			.vertexAttributeDescriptionCount = std::size(attributes),
			.pVertexAttributeDescriptions = attributes,
		};

		// NOTE: Every three vertices make up a triangle,
		//       so our vertex buffer contains a "list of triangles"
		VkPipelineInputAssemblyStateCreateInfo assembly_state_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
			.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		};

		// NOTE: Declare clockwise triangle order as front-facing
		//       Discard triangles that are facing away
		//       Fill triangles, don't draw lines instaed
		VkPipelineRasterizationStateCreateInfo raster_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
			.polygonMode = VK_POLYGON_MODE_FILL,
			.cullMode = VK_CULL_MODE_BACK_BIT,
			.frontFace = VK_FRONT_FACE_CLOCKWISE,
			.lineWidth = 1.0f,
		};

		// NOTE: Use 1 sample per pixel
		VkPipelineMultisampleStateCreateInfo sample_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
			.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
			.sampleShadingEnable = false,
			.minSampleShading = 1.0f,
		};

		VkViewport viewport{
			.x = 0.0f,
			.y = 0.0f,
			.width = static_cast<float>(veekay::app.window_width),
			.height = static_cast<float>(veekay::app.window_height),
			.minDepth = 0.0f,
			.maxDepth = 1.0f,
		};

		VkRect2D scissor{
			.offset = {0, 0},
			.extent = {veekay::app.window_width, veekay::app.window_height},
		};

		// NOTE: Let rasterizer draw on the entire window
		VkPipelineViewportStateCreateInfo viewport_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,

			.viewportCount = 1,
			.pViewports = &viewport,

			.scissorCount = 1,
			.pScissors = &scissor,
		};

		// NOTE: Let rasterizer perform depth-testing and overwrite depth values on condition pass
		VkPipelineDepthStencilStateCreateInfo depth_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
			.depthTestEnable = true,
			.depthWriteEnable = true,
			.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
		};

		// NOTE: Let fragment shader write all the color channels
		VkPipelineColorBlendAttachmentState attachment_info{
			.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
							  VK_COLOR_COMPONENT_G_BIT |
							  VK_COLOR_COMPONENT_B_BIT |
							  VK_COLOR_COMPONENT_A_BIT,
		};

		// NOTE: Let rasterizer just copy resulting pixels onto a buffer, don't blend
		VkPipelineColorBlendStateCreateInfo blend_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,

			.logicOpEnable = false,
			.logicOp = VK_LOGIC_OP_COPY,

			.attachmentCount = 1,
			.pAttachments = &attachment_info
		};

		// NOTE: Declare constant memory region visible to vertex and fragment shaders
		VkPushConstantRange push_constants{
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT |
						  VK_SHADER_STAGE_FRAGMENT_BIT,
			.size = sizeof(ShaderConstants),
		};

		// NOTE: Declare external data sources, only push constants this time
		VkPipelineLayoutCreateInfo layout_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.pushConstantRangeCount = 1,
			.pPushConstantRanges = &push_constants,
		};

		// NOTE: Create pipeline layout
		if (vkCreatePipelineLayout(device, &layout_info,
								   nullptr, &pipeline_layout) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan pipeline layout\n";
			veekay::app.running = false;
			return;
								   }

		VkGraphicsPipelineCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
			.stageCount = 2,
			.pStages = stage_infos,
			.pVertexInputState = &input_state_info,
			.pInputAssemblyState = &assembly_state_info,
			.pViewportState = &viewport_info,
			.pRasterizationState = &raster_info,
			.pMultisampleState = &sample_info,
			.pDepthStencilState = &depth_info,
			.pColorBlendState = &blend_info,
			.layout = pipeline_layout,
			.renderPass = veekay::app.vk_render_pass,
		};

		// NOTE: Create graphics pipeline
		if (vkCreateGraphicsPipelines(device, nullptr,
									  1, &info, nullptr, &pipeline) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan pipeline\n";
			veekay::app.running = false;
			return;
									  }
	}

	std::random_device rd;

	std::mt19937 e2(rd());

	std::uniform_real_distribution<> dist_z(1, 10);
	std::uniform_real_distribution<> dist_xy(-2, 2);
	std::uniform_int_distribution<> dist2(-20, 20);

	constexpr int cone_vert_n = 10000;

	constexpr int starting_cone_n = 10;

	for (int i = 0; i < starting_cone_n; ++i) {
		SceneObject cone{
			.id = i,
			.position = {(float)dist_xy(e2), (float)dist_xy(e2), (float)dist_z(e2)},
		};

		int new_vert_n = cone_vert_n + dist2(e2);

		std::vector<Vector> vertex_coords = getConeVertices(new_vert_n);

		std::vector<uint32_t> indices = getConeIndices(new_vert_n);

		Vertex *vertices = new Vertex[vertex_coords.size()];

		const Vector colors[3] = {
			{1, 0, 0},
			{0, 1, 0},
			{0, 0, 1}
		};

		for (int i = 0; i < vertex_coords.size(); ++i) {
			vertices[i] = {vertex_coords[i], colors[i % 3]};
		}

		cone.mesh.index_count = indices.size();

		cone.mesh.vertex_buffer = createBuffer(sizeof(Vertex) * vertex_coords.size(), vertices, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
		cone.mesh.index_buffer = createBuffer(sizeof(uint32_t) * indices.size(), indices.data(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

		delete[] vertices;

		objects.push_back(cone);
	}

	objects[0].parent = &objects[1];
}

void shutdown() {
	VkDevice& device = veekay::app.vk_device;

	// NOTE: Destroy resources here, do not cause leaks in your program!
	for (const auto &object: objects) {
		destroyBuffer(object.mesh.index_buffer);
		destroyBuffer(object.mesh.vertex_buffer);
	}

	vkDestroyPipeline(device, pipeline, nullptr);
	vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
	vkDestroyShaderModule(device, fragment_shader_module, nullptr);
	vkDestroyShaderModule(device, vertex_shader_module, nullptr);
}

void update(double time) {
	ImGui::Begin("Controls:");

	int i = 0;

	for (auto &object: objects) {
		int old_parent = -1;
		int parent_id = -1;

		if (object.parent) {
			old_parent = object.parent->id;
			parent_id = object.parent->id;
		}

		ImGui::PushID(i);
		ImGui::Text("Object #%d:", i);
		ImGui::InputFloat3("Translation", reinterpret_cast<float*>(&object.position));
		ImGui::InputFloat3("Scale", reinterpret_cast<float*>(&object.scale));
		ImGui::SliderFloat("Rotation", &object.rotation, 0.0f, 2.0f * M_PI);
		ImGui::InputFloat3("Rotation Axis", reinterpret_cast<float*>(&object.rotation_axis));
		ImGui::Checkbox("Spin?", &object.spin);
		ImGui::SliderFloat("Rotation Speed", &object.rotation_speed, 0.0f, 10);
		ImGui::InputInt("Parent ID ", &parent_id);
		ImGui::Separator();
		ImGui::PopID();

		if (old_parent != parent_id && parent_id > -1 && parent_id < objects.size() && parent_id != object.id) {
			object.parent = &objects[parent_id];
		}

		++i;
	}

	ImGui::End();

	dt = time - time_prev;

	// NOTE: Animation code and other runtime variable updates go here

	for (auto &object: objects) {
		if (object.spin) {
			object.rotation += dt * object.rotation_speed;
		}

		object.rotation = fmodf(object.rotation, 2.0f * M_PI);
	}
	time_prev = time;
}

void render(VkCommandBuffer cmd, VkFramebuffer framebuffer) {
	vkResetCommandBuffer(cmd, 0);

	{ // NOTE: Start recording rendering commands
		VkCommandBufferBeginInfo info{
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
		};

		vkBeginCommandBuffer(cmd, &info);
	}

	{ // NOTE: Use current swapchain framebuffer and clear it
		VkClearValue clear_color{.color = {{0.1f, 0.1f, 0.1f, 1.0f}}};
		VkClearValue clear_depth{.depthStencil = {1.0f, 0}};

		VkClearValue clear_values[] = {clear_color, clear_depth};

		VkRenderPassBeginInfo info{
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
			.renderPass = veekay::app.vk_render_pass,
			.framebuffer = framebuffer,
			.renderArea = {
				.extent = {
					veekay::app.window_width,
					veekay::app.window_height
				},
			},
			.clearValueCount = 2,
			.pClearValues = clear_values,
		};

		vkCmdBeginRenderPass(cmd, &info, VK_SUBPASS_CONTENTS_INLINE);
	}

	// NOTE: ShaderConstant updates, vkCmdXXX expected to be here
	{
		// NOTE: Use our new shiny graphics pipeline
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);


		for (auto &object: objects) {
			// NOTE: Use our quad vertex buffer
			VkDeviceSize offset = 0;
			vkCmdBindVertexBuffers(cmd, 0, 1, &object.mesh.vertex_buffer.buffer, &offset);

			// NOTE: Use our quad index buffer
			vkCmdBindIndexBuffer(cmd, object.mesh.index_buffer.buffer, offset, VK_INDEX_TYPE_UINT32);

			Matrix objCumTransform = scaling(object.scale);

			objCumTransform = multiply(objCumTransform, rotation(object.rotation_axis, object.rotation));

			objCumTransform = multiply(objCumTransform, translation(object.position));

			SceneObject *temp = &object;

			while (temp->parent) {
				temp = temp->parent;
				objCumTransform = multiply(objCumTransform, scaling(temp->scale));
				objCumTransform = multiply(objCumTransform, rotation(temp->rotation_axis, temp->rotation));
				objCumTransform = multiply(objCumTransform, translation(temp->position));
			}

			// NOTE: Variables like model_XXX were declared globally
			ShaderConstants constants{
				.projection = projection(
					camera_fov,
					static_cast<float>(veekay::app.window_width) / static_cast<float>(veekay::app.window_height),
					camera_near_plane, camera_far_plane),

				.transform = objCumTransform,

				// .color = model_color,
			};

			// NOTE: Update constant memory with new shader constants
			vkCmdPushConstants(cmd, pipeline_layout,
							   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
							   0, sizeof(ShaderConstants), &constants);

			vkCmdDrawIndexed(cmd, object.mesh.index_count, 1, 0, 0, 0);
		}
	}

	vkCmdEndRenderPass(cmd);
	vkEndCommandBuffer(cmd);
}

} // namespace

int main() {
	return veekay::run({
		.init = initialize,
		.shutdown = shutdown,
		.update = update,
		.render = render,
	});
}
