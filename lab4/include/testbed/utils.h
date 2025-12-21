#ifndef VEEKAY_UTILS_H
#define VEEKAY_UTILS_H

#include <string>
#include <vector>
#include <cstdint>
#include <lodepng.h>
#include <stdexcept>
#include <iostream>

inline veekay::graphics::Texture* loadTexture(VkCommandBuffer cmd, const std::string& path) {
    uint32_t width, height;
    std::vector<uint8_t> pixels;

    if (lodepng::decode(pixels, width, height, path) != 0)
        throw std::runtime_error("Png was not found or error has occurred");

    return new veekay::graphics::Texture(cmd, width, height, VK_FORMAT_R8G8B8A8_UNORM, pixels.data());
}

inline VkSampler createTextureSampler() {
    VkSamplerCreateInfo sampler_info{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_LINEAR,
        .minFilter = VK_FILTER_LINEAR,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .anisotropyEnable = true,
        .maxAnisotropy = 16.0f,
        .minLod = 0.0f,
        .maxLod = VK_LOD_CLAMP_NONE,
    };

    VkSampler sampler;
    if (vkCreateSampler(veekay::app.vk_device, &sampler_info, nullptr, &sampler) != VK_SUCCESS) {
        std::cerr << "Failed to create Vulkan sampler\n";
        return VK_NULL_HANDLE;
    }
    return sampler;
}

#endif //VEEKAY_UTILS_H