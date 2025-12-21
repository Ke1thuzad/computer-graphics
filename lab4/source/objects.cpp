#include <testbed/main.h>

void Material::createMaterialDescriptorSet(VkDescriptorPool material_descriptor_pool, VkDescriptorSetLayout *descriptor_set_layout) {
    VkDescriptorSetAllocateInfo alloc_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = material_descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = descriptor_set_layout,
    };

    if (vkAllocateDescriptorSets(veekay::app.vk_device, &alloc_info, &descriptor_set) != VK_SUCCESS) {
        std::cerr << "Failed to create Vulkan descriptor set\n";
        veekay::app.running = false;
        return;
    }

    VkDescriptorImageInfo albedo_image_info{
        .sampler = sampler,
        .imageView = texture->view,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };

    VkDescriptorImageInfo specular_image_info{
        .sampler = sampler,
        .imageView = specular_texture->view,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };

    VkDescriptorImageInfo emissive_image_info{
        .sampler = sampler,
        .imageView = emissive_texture->view,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };

    VkWriteDescriptorSet write_infos[3] = {
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo = &albedo_image_info,
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set,
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo = &specular_image_info,
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set,
            .dstBinding = 2,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo = &emissive_image_info,
        }
    };

    vkUpdateDescriptorSets(veekay::app.vk_device, 3, write_infos, 0, nullptr);
}

veekay::mat4 Transform::matrix() const {
    auto t = veekay::mat4::translation(position);

    auto r = veekay::mat4::rotation({1, 0, 0}, rotation.y)
             * veekay::mat4::rotation({0, 1, 0}, rotation.x)
             * veekay::mat4::rotation({0, 0, 1}, rotation.z);

    auto s = veekay::mat4::scaling(scale);

    return s * r * t;
}

veekay::mat4 Camera::view() const {
    auto t = veekay::mat4::translation(-position);

    auto rotX = veekay::mat4::rotation({1.0f, 0.0f, 0.0f}, rotation.x);
    auto rotY = veekay::mat4::rotation({0.0f, 1.0f, 0.0f}, rotation.y);
    auto rotZ = veekay::mat4::rotation({0.0f, 0.0f, 1.0f}, rotation.z);

    auto r = rotY * rotX * rotZ;

    return t * r;
}

veekay::mat4 Camera::look_at(veekay::vec3 at) const {
    const veekay::vec3 forward = veekay::vec3::normalized(position - at);

    veekay::vec3 world_up = {0, 1, 0};

    veekay::vec3 right = veekay::vec3::normalized(veekay::vec3::cross(forward, world_up));

    veekay::vec3 up = veekay::vec3::normalized(veekay::vec3::cross(right, forward));

    const veekay::mat4 basis = {
        right.x, up.x, -forward.x, 0,
        right.y, up.y, -forward.y, 0,
        right.z, up.z, -forward.z, 0,
        0, 0, 0, 1
    };

    return veekay::mat4::translation(-position) * basis;
}

veekay::mat4 Camera::view_projection(const float aspect_ratio, const veekay::mat4 &view) const {
    const auto projection = veekay::mat4::projection(fov, aspect_ratio, near_plane, far_plane);

    return view * projection;
}
