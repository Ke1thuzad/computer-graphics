#define _USE_MATH_DEFINES

#include "../include/testbed/main.h"

namespace {
    // Scene objects
    inline namespace {
        Camera camera{
            .position = {0.0f, -0.5f, -3.0f}
        };

        std::vector<Model> models;
        std::vector<PointLight> point_lights{};
        std::vector<SpotLight> spot_lights{};
    }

    // Imgui objects
    inline namespace {
        bool is_look_at = false;
    }

    inline namespace {
        VkShaderModule vertex_shader_module;
        VkShaderModule fragment_shader_module;

        VkDescriptorPool descriptor_pool;
        VkDescriptorPool material_descriptor_pool;
        VkDescriptorSetLayout descriptor_set_layouts[2];
        VkDescriptorSet descriptor_set;

        VkPipelineLayout pipeline_layout;
        VkPipeline pipeline;

        veekay::graphics::Buffer *scene_uniforms_buffer;
        veekay::graphics::Buffer *model_uniforms_buffer;
        veekay::graphics::Buffer *point_lights_buffer;
        veekay::graphics::Buffer *spot_lights_buffer;

        Mesh plane_mesh;
        Mesh cube_mesh;

        veekay::graphics::Texture *missing_texture;
        VkSampler missing_texture_sampler;

        veekay::graphics::Texture* white_texture = nullptr;
        veekay::graphics::Texture* black_texture = nullptr;
        VkSampler white_texture_sampler = VK_NULL_HANDLE;

        std::shared_ptr<Material> default_material;
        std::unordered_map<std::string, std::shared_ptr<Material>> materials_map;
    }

    float toRadians(float degrees) {
        return degrees * static_cast<float>(M_PI) / 180.0f;
    }

    VkShaderModule loadShaderModule(const char *path) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        size_t size = file.tellg();
        std::vector<uint32_t> buffer(size / sizeof(uint32_t));
        file.seekg(0);
        file.read(reinterpret_cast<char *>(buffer.data()), size);
        file.close();

        VkShaderModuleCreateInfo info{
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = size,
            .pCode = buffer.data(),
        };

        VkShaderModule result;
        if (vkCreateShaderModule(veekay::app.vk_device, &info, nullptr, &result) != VK_SUCCESS) {
            return nullptr;
        }

        return result;
    }

    std::shared_ptr<Material> createTextureMaterial(VkCommandBuffer cmd, const std::string& base_name,
                                                    const std::string& albedo_path,
                                                    const std::string& specular_path = "",
                                                    const std::string& emissive_path = "") {

    if (materials_map.contains(base_name)) {
        std::cerr << "This material name (" << base_name << ") already exists, try another" << std::endl;
        return materials_map[base_name];
    }

    auto material = std::make_shared<Material>();

    material->texture = loadTexture(cmd, albedo_path);
    if (!material->texture) {
        material->texture = white_texture;
    }

    if (!specular_path.empty() && std::filesystem::exists(specular_path)) {
        material->specular_texture = loadTexture(cmd, specular_path);
    }
    if (!material->specular_texture) {
        material->specular_texture = white_texture;
    }

    if (!emissive_path.empty() && std::filesystem::exists(emissive_path)) {
        material->emissive_texture = loadTexture(cmd, emissive_path);
    }
    if (!material->emissive_texture) {
        material->emissive_texture = black_texture;
    }

    material->sampler = createTextureSampler();
    if (material->sampler == VK_NULL_HANDLE) {
        material->sampler = white_texture_sampler;
    }

    material->createMaterialDescriptorSet(material_descriptor_pool, &descriptor_set_layouts[1]);

    materials_map[base_name] = material;

    return material;
}

    std::shared_ptr<Material> createColorMaterial(const veekay::vec3& albedo, const veekay::vec3& specular, float shininess, const std::string& name = "") {
        auto material = std::make_shared<Material>();
        material->albedo_color = albedo;
        material->specular_color = specular;
        material->shininess = shininess;
        material->texture = white_texture;
        material->specular_texture = white_texture;
        material->emissive_texture = black_texture;
        material->sampler = white_texture_sampler;

        VkDescriptorSetAllocateInfo alloc_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = material_descriptor_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &descriptor_set_layouts[1],
        };

        if (vkAllocateDescriptorSets(veekay::app.vk_device, &alloc_info, &material->descriptor_set) != VK_SUCCESS) {
            std::cerr << "Failed to create Vulkan descriptor set\n";
            veekay::app.running = false;
            return material;
        }

        VkDescriptorImageInfo albedo_image_info{
            .sampler = material->sampler,
            .imageView = material->texture->view,
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        };

        VkDescriptorImageInfo specular_image_info{
            .sampler = material->sampler,
            .imageView = material->specular_texture->view,
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        };

        VkDescriptorImageInfo emissive_image_info{
            .sampler = material->sampler,
            .imageView = material->emissive_texture->view,
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        };

        VkWriteDescriptorSet write_infos[3] = {
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = material->descriptor_set,
                .dstBinding = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo = &albedo_image_info,
            },
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = material->descriptor_set,
                .dstBinding = 1,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo = &specular_image_info,
            },
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = material->descriptor_set,
                .dstBinding = 2,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo = &emissive_image_info,
            }
        };

        vkUpdateDescriptorSets(veekay::app.vk_device, 3, write_infos, 0, nullptr);

        if (!name.empty()) {
            materials_map[name] = material;
        }

        return material;
    }

    std::shared_ptr<Material> getMaterial(const std::string& name) {
        auto it = materials_map.find(name);
        if (it != materials_map.end()) {
            return it->second;
        }

        std::cerr << "Material not found: " << name << ", using default material\n";
        return default_material;
    }

    void initialize(VkCommandBuffer cmd) {
        VkDevice &device = veekay::app.vk_device;
        VkPhysicalDevice &physical_device = veekay::app.vk_physical_device;

        {
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

            stage_infos[0] = VkPipelineShaderStageCreateInfo{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_VERTEX_BIT,
                .module = vertex_shader_module,
                .pName = "main",
            };

            stage_infos[1] = VkPipelineShaderStageCreateInfo{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
                .module = fragment_shader_module,
                .pName = "main",
            };

            VkVertexInputBindingDescription buffer_binding{
                .binding = 0,
                .stride = sizeof(Vertex),
                .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
            };

            VkVertexInputAttributeDescription attributes[] = {
                {
                    .location = 0,
                    .binding = 0,
                    .format = VK_FORMAT_R32G32B32_SFLOAT,
                    .offset = offsetof(Vertex, position),
                },
                {
                    .location = 1,
                    .binding = 0,
                    .format = VK_FORMAT_R32G32B32_SFLOAT,
                    .offset = offsetof(Vertex, normal),
                },
                {
                    .location = 2,
                    .binding = 0,
                    .format = VK_FORMAT_R32G32_SFLOAT,
                    .offset = offsetof(Vertex, uv),
                },
            };

            VkPipelineVertexInputStateCreateInfo input_state_info{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                .vertexBindingDescriptionCount = 1,
                .pVertexBindingDescriptions = &buffer_binding,
                .vertexAttributeDescriptionCount = std::size(attributes),
                .pVertexAttributeDescriptions = attributes,
            };

            VkPipelineInputAssemblyStateCreateInfo assembly_state_info{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            };

            VkPipelineRasterizationStateCreateInfo raster_info{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                .polygonMode = VK_POLYGON_MODE_FILL,
                .cullMode = VK_CULL_MODE_BACK_BIT,
                .frontFace = VK_FRONT_FACE_CLOCKWISE,
                .lineWidth = 1.0f,
            };

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

            VkPipelineViewportStateCreateInfo viewport_info{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,

                .viewportCount = 1,
                .pViewports = &viewport,

                .scissorCount = 1,
                .pScissors = &scissor,
            };

            VkPipelineDepthStencilStateCreateInfo depth_info{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
                .depthTestEnable = true,
                .depthWriteEnable = true,
                .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
            };

            VkPipelineColorBlendAttachmentState attachment_info{
                .blendEnable = true,
                .srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
                .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
                .colorBlendOp = VK_BLEND_OP_ADD,
                .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
                .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
                .alphaBlendOp = VK_BLEND_OP_ADD,
                .colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
                                  VK_COLOR_COMPONENT_G_BIT |
                                  VK_COLOR_COMPONENT_B_BIT |
                                  VK_COLOR_COMPONENT_A_BIT,
            };

            VkPipelineColorBlendStateCreateInfo blend_info{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,

                .logicOpEnable = false,
                .logicOp = VK_LOGIC_OP_COPY,

                .attachmentCount = 1,
                .pAttachments = &attachment_info
            };

            {
                {
                    VkDescriptorPoolSize pools[] = {
                        {
                            .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                            .descriptorCount = 8,
                        },
                        {
                            .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                            .descriptorCount = 8,
                        },
                        {
                            .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                            .descriptorCount = 8,
                        }
                    };

                    VkDescriptorPoolCreateInfo info{
                        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                        .maxSets = 1,
                        .poolSizeCount = std::size(pools),
                        .pPoolSizes = pools,
                    };

                    if (vkCreateDescriptorPool(device, &info, nullptr,
                                               &descriptor_pool) != VK_SUCCESS) {
                        std::cerr << "Failed to create Vulkan descriptor pool\n";
                        veekay::app.running = false;
                        return;
                    }
                }

                {
                    VkDescriptorPoolSize pools[] = {
                        {
                            .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                            .descriptorCount = max_textures * 3,
                        }
                    };

                    VkDescriptorPoolCreateInfo info{
                        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                        .maxSets = max_textures,
                        .poolSizeCount = std::size(pools),
                        .pPoolSizes = pools,
                    };

                    if (vkCreateDescriptorPool(device, &info, nullptr,
                                               &material_descriptor_pool) != VK_SUCCESS) {
                        std::cerr << "Failed to create Vulkan descriptor pool\n";
                        veekay::app.running = false;
                        return;
                    }
                }
            }

            {
                VkDescriptorSetLayoutBinding bindings[] = {
                    {
                        .binding = 0,
                        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                        .descriptorCount = 1,
                        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                    },
                    {
                        .binding = 1,
                        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                        .descriptorCount = 1,
                        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                    },
                    {
                        .binding = 2,
                        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                        .descriptorCount = 1,
                        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT
                    },
                    {
                        .binding = 3,
                        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                        .descriptorCount = 1,
                        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT
                    }
                };


                VkDescriptorSetLayoutCreateInfo info{
                    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                    .bindingCount = std::size(bindings),
                    .pBindings = bindings,
                };

                if (vkCreateDescriptorSetLayout(device, &info, nullptr,
                                                &descriptor_set_layouts[0]) != VK_SUCCESS) {
                    std::cerr << "Failed to create Vulkan descriptor set layout\n";
                    veekay::app.running = false;
                    return;
                }

                VkDescriptorSetLayoutBinding material_bindings[] = {
                    {
                        .binding = 0,
                        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        .descriptorCount = 1,
                        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT
                    },
                    {
                        .binding = 1,
                        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        .descriptorCount = 1,
                        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT
                    },
                    {
                        .binding = 2,
                        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        .descriptorCount = 1,
                        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT
                    }
                };

                VkDescriptorSetLayoutCreateInfo info_materials{
                    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                    .bindingCount = std::size(material_bindings),
                    .pBindings = material_bindings,
                };

                if (vkCreateDescriptorSetLayout(device, &info_materials, nullptr,
                                                &descriptor_set_layouts[1]) != VK_SUCCESS) {
                    std::cerr << "Failed to create Vulkan descriptor set layout\n";
                    veekay::app.running = false;
                    return;
                }
            }

            {
                VkDescriptorSetAllocateInfo scene_info{
                    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                    .descriptorPool = descriptor_pool,
                    .descriptorSetCount = 1,
                    .pSetLayouts = &descriptor_set_layouts[0],
                };

                if (vkAllocateDescriptorSets(device, &scene_info, &descriptor_set) != VK_SUCCESS) {
                    std::cerr << "Failed to create Vulkan descriptor set\n";
                    veekay::app.running = false;
                    return;
                }
            }

            VkPipelineLayoutCreateInfo layout_info{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                .setLayoutCount = 2,
                .pSetLayouts = descriptor_set_layouts,
            };

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

            if (vkCreateGraphicsPipelines(device, nullptr,
                                          1, &info, nullptr, &pipeline) != VK_SUCCESS) {
                std::cerr << "Failed to create Vulkan pipeline\n";
                veekay::app.running = false;
                return;
            }
        }

        scene_uniforms_buffer = new veekay::graphics::Buffer(
            sizeof(SceneUniforms),
            nullptr,
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

        model_uniforms_buffer = new veekay::graphics::Buffer(
            max_models * veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms)),
            nullptr,
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

        point_lights_buffer = new veekay::graphics::Buffer(
            max_point_lights * sizeof(PointLight),
            nullptr,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

        spot_lights_buffer = new veekay::graphics::Buffer(
            max_spot_lights * sizeof(SpotLight),
            nullptr,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

        {
            VkSamplerCreateInfo info{
                .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            };

            if (vkCreateSampler(device, &info, nullptr, &missing_texture_sampler) != VK_SUCCESS) {
                std::cerr << "Failed to create Vulkan texture sampler\n";
                veekay::app.running = false;
                return;
            }

            uint32_t pixels[] = {
                0xff000000, 0xffff00ff,
                0xffff00ff, 0xff000000,
            };

            missing_texture = new veekay::graphics::Texture(cmd, 2, 2,
                                                            VK_FORMAT_B8G8R8A8_UNORM,
                                                            pixels);
        }

        {
            uint32_t white_pixel = 0xFFFFFFFF;
            white_texture = new veekay::graphics::Texture(cmd, 1, 1,
                                                         VK_FORMAT_B8G8R8A8_UNORM,
                                                         &white_pixel);

            uint32_t black_pixel = 0xFF000000;
            black_texture = new veekay::graphics::Texture(cmd, 1, 1,
                                                         VK_FORMAT_B8G8R8A8_UNORM,
                                                         &black_pixel);

            VkSamplerCreateInfo sampler_info{
                .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                .magFilter = VK_FILTER_LINEAR,
                .minFilter = VK_FILTER_LINEAR,
                .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
                .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
                .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
                .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
                .anisotropyEnable = false,
                .minLod = 0.0f,
                .maxLod = VK_LOD_CLAMP_NONE,
            };

            if (vkCreateSampler(device, &sampler_info, nullptr, &white_texture_sampler) != VK_SUCCESS) {
                std::cerr << "Failed to create white texture sampler\n";
                veekay::app.running = false;
                return;
            }
        }

        {
            VkDescriptorBufferInfo buffer_infos[] = {
                {
                    .buffer = scene_uniforms_buffer->buffer,
                    .offset = 0,
                    .range = sizeof(SceneUniforms),
                },
                {
                    .buffer = model_uniforms_buffer->buffer,
                    .offset = 0,
                    .range = sizeof(ModelUniforms),
                },
                {
                    .buffer = point_lights_buffer->buffer,
                    .offset = 0,
                    .range = max_point_lights * sizeof(PointLight),
                },
                {
                    .buffer = spot_lights_buffer->buffer,
                    .offset = 0,
                    .range = max_spot_lights * sizeof(SpotLight),
                }
            };

            VkWriteDescriptorSet write_infos[] = {
                {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = descriptor_set,
                    .dstBinding = 0,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    .pBufferInfo = &buffer_infos[0],
                },
                {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = descriptor_set,
                    .dstBinding = 1,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                    .pBufferInfo = &buffer_infos[1],
                },
                {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = descriptor_set,
                    .dstBinding = 2,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .pBufferInfo = &buffer_infos[2],
                },
                {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = descriptor_set,
                    .dstBinding = 3,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .pBufferInfo = &buffer_infos[3],
                },
            };

            vkUpdateDescriptorSets(device, std::size(write_infos),
                                   write_infos, 0, nullptr);
        }

        {
            default_material = std::make_shared<Material>();
            default_material->albedo_color = {1.0f, 1.0f, 1.0f};
            default_material->specular_color = {1.0f, 1.0f, 1.0f};
            default_material->shininess = 100.0f;
            default_material->texture = white_texture;
            default_material->specular_texture = white_texture;
            default_material->emissive_texture = black_texture;
            default_material->sampler = white_texture_sampler;

            VkDescriptorSetAllocateInfo alloc_info{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .descriptorPool = material_descriptor_pool,
                .descriptorSetCount = 1,
                .pSetLayouts = &descriptor_set_layouts[1],
            };

            if (vkAllocateDescriptorSets(device, &alloc_info, &default_material->descriptor_set) != VK_SUCCESS) {
                std::cerr << "Failed to create default Vulkan descriptor set\n";
                veekay::app.running = false;
                return;
            }

            VkDescriptorImageInfo albedo_image_info{
                .sampler = default_material->sampler,
                .imageView = default_material->texture->view,
                .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            };

            VkDescriptorImageInfo specular_image_info{
                .sampler = default_material->sampler,
                .imageView = default_material->specular_texture->view,
                .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            };

            VkDescriptorImageInfo emissive_image_info{
                .sampler = default_material->sampler,
                .imageView = default_material->emissive_texture->view,
                .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            };

            VkWriteDescriptorSet write_infos[3] = {
                {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = default_material->descriptor_set,
                    .dstBinding = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .pImageInfo = &albedo_image_info,
                },
                {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = default_material->descriptor_set,
                    .dstBinding = 1,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .pImageInfo = &specular_image_info,
                },
                {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = default_material->descriptor_set,
                    .dstBinding = 2,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .pImageInfo = &emissive_image_info,
                }
            };

            vkUpdateDescriptorSets(device, 3, write_infos, 0, nullptr);

            materials_map["default"] = default_material;
            materials_map["white"] = default_material;
        }

        {
            std::unordered_map<std::string, std::tuple<std::string, std::string, std::string>> texture_files;

            for (const auto &dirEntry: std::filesystem::recursive_directory_iterator("./assets/textures")) {
                if (!dirEntry.is_regular_file() || dirEntry.path().extension() != ".png")
                    continue;

                std::string filename = dirEntry.path().stem().string();
                std::string path = dirEntry.path().string();

                if (filename.find("_specular") != std::string::npos) {
                    std::string base_name = filename.substr(0, filename.find("_specular"));
                    std::get<1>(texture_files[base_name]) = path;
                } else if (filename.find("_emissive") != std::string::npos) {
                    std::string base_name = filename.substr(0, filename.find("_emissive"));
                    std::get<2>(texture_files[base_name]) = path;
                } else {
                    std::get<0>(texture_files[filename]) = path;
                }
            }

            for (const auto& [base_name, paths] : texture_files) {
                const auto& [albedo_path, specular_path, emissive_path] = paths;
                if (!albedo_path.empty()) {
                    createTextureMaterial(cmd, base_name, albedo_path, specular_path, emissive_path);
                }
            }
        }

        {
            std::vector<Vertex> vertices = {
                {{-5.0f, 0.0f, 5.0f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
                {{5.0f, 0.0f, 5.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}},
                {{5.0f, 0.0f, -5.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}},
                {{-5.0f, 0.0f, -5.0f}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}},
            };

            std::vector<uint32_t> indices = {
                0, 1, 2, 2, 3, 0
            };

            plane_mesh.vertex_buffer = new veekay::graphics::Buffer(
                vertices.size() * sizeof(Vertex), vertices.data(),
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

            plane_mesh.index_buffer = new veekay::graphics::Buffer(
                indices.size() * sizeof(uint32_t), indices.data(),
                VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

            plane_mesh.indices = static_cast<uint32_t>(indices.size());
        }

        {
            std::vector<Vertex> vertices = {
                {{-0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 0.0f}},
                {{+0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 0.0f}},
                {{+0.5f, +0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 1.0f}},
                {{-0.5f, +0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f}},

                {{+0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
                {{+0.5f, -0.5f, +0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
                {{+0.5f, +0.5f, +0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
                {{+0.5f, +0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},

                {{+0.5f, -0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}},
                {{-0.5f, -0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}},
                {{-0.5f, +0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
                {{+0.5f, +0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},

                {{-0.5f, -0.5f, +0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
                {{-0.5f, -0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
                {{-0.5f, +0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
                {{-0.5f, +0.5f, +0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},

                {{-0.5f, -0.5f, +0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
                {{+0.5f, -0.5f, +0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}},
                {{+0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}},
                {{-0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}},

                {{-0.5f, +0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
                {{+0.5f, +0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
                {{+0.5f, +0.5f, +0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}},
                {{-0.5f, +0.5f, +0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}},
            };

            std::vector<uint32_t> indices = {
                0, 1, 2, 2, 3, 0,
                4, 5, 6, 6, 7, 4,
                8, 9, 10, 10, 11, 8,
                12, 13, 14, 14, 15, 12,
                16, 17, 18, 18, 19, 16,
                20, 21, 22, 22, 23, 20,
            };

            cube_mesh.vertex_buffer = new veekay::graphics::Buffer(
                vertices.size() * sizeof(Vertex), vertices.data(),
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

            cube_mesh.index_buffer = new veekay::graphics::Buffer(
                indices.size() * sizeof(uint32_t), indices.data(),
                VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

            cube_mesh.indices = static_cast<uint32_t>(indices.size());
        }

        // Создаем экземпляры материалов
        {
            createColorMaterial({0.5f, 0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}, 500.0f, "grey");
            createColorMaterial({1.0f, 0.5f, 0.1f}, {1.0f, 1.0f, 1.0f}, 10000.0f, "orange");
            createColorMaterial({0.6f, 1.0f, 0.2f}, {1.0f, 1.0f, 1.0f}, 10.0f, "green");
            createColorMaterial({0.3f, 0.1f, 1.0f}, {1.0f, 1.0f, 1.0f}, 25.0f, "blue");

            // createTextureMaterial(cmd, "mandarinka_spec", "./assets/textures/mandarinka.png", "./assets/textures/mandarinka_emissive.png");
        }

        models.emplace_back(Model{
            .mesh = plane_mesh,
            .transform = Transform{},
            .material = getMaterial("grey")
        });

        models.emplace_back(Model{
            .mesh = plane_mesh,
            .transform = Transform{
                .position = {0, -5, 5},
                .rotation = {0, toRadians(90), 0},
            },
            .material = getMaterial("grey")
        });

        models.emplace_back(Model{
            .mesh = plane_mesh,
            .transform = Transform{
                .position = {5, -5, 0},
                .rotation = {toRadians(90), 0, -toRadians(90)},
            },
            .material = getMaterial("grey")
        });

        models.emplace_back(Model{
            .mesh = plane_mesh,
            .transform = Transform{
                .position = {-5, -5, 0},
                .rotation = {-toRadians(90), 0, toRadians(90)},
            },
            .material = getMaterial("grey")
        });

        models.emplace_back(Model{
            .mesh = cube_mesh,
            .transform = Transform{
                .position = {-2.0f, -0.5f, -1.5f},
                .scale = {0.5f, 0.9f, 0.9f},
            },
            .material = getMaterial("orange")
        });

        models.emplace_back(Model{
            .mesh = cube_mesh,
            .transform = Transform{
                .position = {1.5f, -0.5f, 0.5f},
            },
            .material = getMaterial("green")
        });

        models.emplace_back(Model{
            .mesh = cube_mesh,
            .transform = Transform{
                .position = {0.0f, -2.5f, 1.0f},
                .scale = {2, 1, 1},
            },
            .material = getMaterial("blue")
        });

        point_lights.emplace_back(PointLight
            ({3, -3.0f, 2}, {0.81f, 0.42f, 0.15f}, 5)
        );

        spot_lights.emplace_back(SpotLight
            (camera.position, {0.81f, 0.42f, 0.15f}, {0, 0, 1}, 10, 0.91f, 0.82f)
        );

        // getMaterial("mandarinka")->shininess = 5000;

        models[0].material = getMaterial("angry");
        models[1].material = getMaterial("lenna");
        models[2].material = getMaterial("okak");
        // models[3].material = getMaterial("mandarinka");
        models[4].material = getMaterial("hehe");
    }

    void shutdown() {
        VkDevice &device = veekay::app.vk_device;

        vkDeviceWaitIdle(device);

        vkDestroySampler(device, missing_texture_sampler, nullptr);
        vkDestroySampler(device, white_texture_sampler, nullptr);
        delete missing_texture;
        delete white_texture;
        delete black_texture;

        for (const auto& [name, material] : materials_map) {
            if (material->texture != white_texture && material->texture != black_texture && material->texture != nullptr)
                delete material->texture;

            if (material->specular_texture != white_texture && material->specular_texture != black_texture && material->specular_texture != nullptr)
                delete material->specular_texture;

            if (material->emissive_texture != white_texture && material->emissive_texture != black_texture && material->emissive_texture != nullptr)
                delete material->emissive_texture;

            if (material->sampler != white_texture_sampler && material->sampler != VK_NULL_HANDLE)
                vkDestroySampler(device, material->sampler, nullptr);
        }
        materials_map.clear();

        delete cube_mesh.index_buffer;
        delete cube_mesh.vertex_buffer;

        delete plane_mesh.index_buffer;
        delete plane_mesh.vertex_buffer;

        delete model_uniforms_buffer;
        delete scene_uniforms_buffer;
        delete point_lights_buffer;
        delete spot_lights_buffer;

        vkDestroyDescriptorSetLayout(device, descriptor_set_layouts[0], nullptr);
        vkDestroyDescriptorSetLayout(device, descriptor_set_layouts[1], nullptr);
        vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
        vkDestroyDescriptorPool(device, material_descriptor_pool, nullptr);

        vkDestroyPipeline(device, pipeline, nullptr);
        vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
        vkDestroyShaderModule(device, fragment_shader_module, nullptr);
        vkDestroyShaderModule(device, vertex_shader_module, nullptr);
    }

    void update(double time) {
        ImGui::Begin("Lights Control");

        ImGui::Checkbox("Look At", &is_look_at);

        if (ImGui::TreeNodeEx("Point Lights", ImGuiTreeNodeFlags_DefaultOpen)) {
            for (int i = 0; i < point_lights.size(); ++i) {
                auto &light = point_lights[i];

                ImGui::PushID(&light);

                if (ImGui::TreeNodeEx(&light, ImGuiTreeNodeFlags_DefaultOpen, "Point Light %d", i)) {
                    ImGui::InputFloat3("Position", light.position.elements);
                    ImGui::ColorEdit3("Color", light.color.elements);
                    ImGui::InputFloat("Radius", &light.radius);

                    if (ImGui::Button("Delete")) {
                        point_lights.erase(point_lights.begin() + i);
                        ImGui::TreePop();
                        ImGui::PopID();
                        break;
                    }

                    ImGui::TreePop();
                }
                ImGui::PopID();
            }

            if (point_lights.size() < max_point_lights && ImGui::Button("Add Point Light")) {
                point_lights.emplace_back();
            }

            ImGui::TreePop();
        }

        if (ImGui::TreeNodeEx("Spot Lights", ImGuiTreeNodeFlags_DefaultOpen)) {
            for (int i = 0; i < spot_lights.size(); ++i) {
                auto &light = spot_lights[i];
                ImGui::PushID(&light);

                if (ImGui::TreeNodeEx(&light, ImGuiTreeNodeFlags_DefaultOpen, "Spot Light %d", i)) {
                    ImGui::InputFloat3("Position", light.position.elements);
                    ImGui::ColorEdit3("Color", light.color.elements);
                    ImGui::InputFloat3("Direction", light.direction.elements);
                    ImGui::InputFloat("Radius", &light.radius);
                    ImGui::InputFloat("Cutoff angle", &light.angle);
                    ImGui::InputFloat("Outer Cutoff angle", &light.outer_angle);

                    if (ImGui::Button("Delete")) {
                        spot_lights.erase(spot_lights.begin() + i);
                        ImGui::TreePop();
                        ImGui::PopID();
                        break;
                    }

                    ImGui::TreePop();
                }
                ImGui::PopID();
            }

            if (spot_lights.size() < max_spot_lights && ImGui::Button("Add Spot Light")) {
                spot_lights.emplace_back();
            }

            ImGui::TreePop();
        }

        ImGui::End();

        if (!ImGui::IsWindowHovered() && !ImGui::IsWindowFocused()) {
            using namespace veekay::input;

            if (mouse::isButtonDown(mouse::Button::left)) {
                auto move_delta = mouse::cursorDelta();

                camera.rotation.x += move_delta.y / 360.0f;
                camera.rotation.y += -move_delta.x / 360.0f;

                camera.rotation.x = std::clamp(camera.rotation.x, -static_cast<float>(M_PI_2),
                                               static_cast<float>(M_PI_2));

                auto view = is_look_at ? camera.look_at({0, 0, 0}) : camera.view();

                veekay::vec3 right = veekay::vec3::normalized({view[0][0], view[1][0], view[2][0]});
                veekay::vec3 up = veekay::vec3::normalized({view[0][1], view[1][1], view[2][1]});
                veekay::vec3 front = veekay::vec3::normalized({view[0][2], view[1][2], view[2][2]});

                if (keyboard::isKeyDown(keyboard::Key::w))
                    camera.position += front * 0.1f;

                if (keyboard::isKeyDown(keyboard::Key::s))
                    camera.position -= front * 0.1f;

                if (keyboard::isKeyDown(keyboard::Key::d))
                    camera.position += right * 0.1f;

                if (keyboard::isKeyDown(keyboard::Key::a))
                    camera.position -= right * 0.1f;

                if (keyboard::isKeyDown(keyboard::Key::q))
                    camera.position += up * 0.1f;

                if (keyboard::isKeyDown(keyboard::Key::z))
                    camera.position -= up * 0.1f;
            }
        }

        veekay::mat4 view_mat{};

        if (!is_look_at) {
            view_mat = camera.view();
        } else {
            view_mat = camera.look_at({0, 0, 0});
        }

        const float aspect_ratio = static_cast<float>(veekay::app.window_width) / static_cast<float>(veekay::app.window_height);
        SceneUniforms scene_uniforms{
            .view_projection = camera.view_projection(aspect_ratio, view_mat),
            .view_position = camera.position,
            .ambient_light_intensity = {0.075f, 0.075f, 0.075f},
            .sun_light_direction = {0.2f, 0.4f, 0.3f},
            .sun_light_color = {1, 1, 1},
            .point_lights_count = static_cast<uint32_t>(point_lights.size()),
            .spot_lights_count = static_cast<uint32_t>(spot_lights.size()),
        };

        std::vector<ModelUniforms> model_uniforms(models.size());
        for (size_t i = 0, n = models.size(); i < n; ++i) {
            const Model &model = models[i];
            ModelUniforms &uniforms = model_uniforms[i];

            uniforms.model = model.transform.matrix();
            uniforms.material.albedo_color = model.material->albedo_color;
            uniforms.material.specular_color = model.material->specular_color;
            uniforms.material.shininess = model.material->shininess;
        }

        *static_cast<SceneUniforms *>(scene_uniforms_buffer->mapped_region) = scene_uniforms;

        const size_t alignment = veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));

        for (size_t i = 0, n = model_uniforms.size(); i < n; ++i) {
            const ModelUniforms &uniforms = model_uniforms[i];

            char *const pointer = static_cast<char *>(model_uniforms_buffer->mapped_region) + i * alignment;
            *reinterpret_cast<ModelUniforms *>(pointer) = uniforms;
        }

        std::ranges::copy(point_lights,
                          static_cast<PointLight *>(point_lights_buffer->mapped_region));

        std::ranges::copy(spot_lights,
                          static_cast<SpotLight *>(spot_lights_buffer->mapped_region));
    }

    void render(VkCommandBuffer cmd, VkFramebuffer framebuffer) {
        vkResetCommandBuffer(cmd, 0);

        {
            VkCommandBufferBeginInfo info{
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            };

            vkBeginCommandBuffer(cmd, &info);
        }

        {
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

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        VkDeviceSize zero_offset = 0;

        VkBuffer current_vertex_buffer = VK_NULL_HANDLE;
        VkBuffer current_index_buffer = VK_NULL_HANDLE;

        const size_t model_uniforms_alignment =
                veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));

        for (size_t i = 0, n = models.size(); i < n; ++i) {
            const Model &model = models[i];
            const Mesh &mesh = model.mesh;

            if (current_vertex_buffer != mesh.vertex_buffer->buffer) {
                current_vertex_buffer = mesh.vertex_buffer->buffer;
                vkCmdBindVertexBuffers(cmd, 0, 1, &current_vertex_buffer, &zero_offset);
            }

            if (current_index_buffer != mesh.index_buffer->buffer) {
                current_index_buffer = mesh.index_buffer->buffer;
                vkCmdBindIndexBuffer(cmd, current_index_buffer, zero_offset, VK_INDEX_TYPE_UINT32);
            }

            VkDescriptorSet descriptor_sets[2] = {
                descriptor_set,
                model.material->descriptor_set
            };

            uint32_t offset = i * model_uniforms_alignment;
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
                                    0, 2, descriptor_sets, 1, &offset);

            vkCmdDrawIndexed(cmd, mesh.indices, 1, 0, 0, 0);
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