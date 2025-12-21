#ifndef VEEKAY_OBJECTS_H
#define VEEKAY_OBJECTS_H

#include <memory>

struct Vertex {
    veekay::vec3 position;
    veekay::vec3 normal;
    veekay::vec2 uv;
};

struct Material {
    veekay::vec3 albedo_color = {1, 1, 1};
    veekay::vec3 specular_color = {1, 1, 1};
    float shininess = 100;

    veekay::graphics::Texture *texture = nullptr;
    veekay::graphics::Texture *specular_texture = nullptr;
    veekay::graphics::Texture *emissive_texture = nullptr;
    VkSampler sampler = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;

    void createMaterialDescriptorSet(VkDescriptorPool material_descriptor_pool, VkDescriptorSetLayout *descriptor_set_layouts);
};

struct Mesh {
    veekay::graphics::Buffer *vertex_buffer;
    veekay::graphics::Buffer *index_buffer;
    uint32_t indices;
};

struct Transform {
    veekay::vec3 position = {};
    veekay::vec3 scale = {1.0f, 1.0f, 1.0f};
    veekay::vec3 rotation = {};

    veekay::mat4 matrix() const;
};

struct Model {
    Mesh mesh;
    Transform transform;
    std::shared_ptr<Material> material;
};

struct PointLight {
    veekay::vec3 position;
    float radius;
    veekay::vec3 color;
    float _pad0;

    PointLight() : position({0, 0, 0}), radius(5), color({1, 1, 1}) {
    }

    PointLight(veekay::vec3 position, veekay::vec3 color, float radius) : position(position), color(color),
                                                                          radius(radius) {
    }
};

struct SpotLight {
    veekay::vec3 position;
    float radius;
    veekay::vec3 color;
    float angle;
    veekay::vec3 direction;
    float outer_angle;

    SpotLight() : position({0, 0, 0}), radius(5), color({1, 1, 1}), angle(0.9f), direction({0, 0, 1}),
                  outer_angle(0.81f) {
    }

    SpotLight(veekay::vec3 position, veekay::vec3 color, veekay::vec3 direction, float radius, float angle,
              float outer_angle) : position(position), color(color), direction(direction), radius(radius),
                                   angle(angle), outer_angle(outer_angle) {
    }
};

struct Camera {
    constexpr static float default_fov = 60.0f;
    constexpr static float default_near_plane = 0.01f;
    constexpr static float default_far_plane = 100.0f;

    veekay::vec3 position = {};
    veekay::vec3 rotation = {};

    float fov = default_fov;
    float near_plane = default_near_plane;
    float far_plane = default_far_plane;

    veekay::mat4 view() const;

    veekay::mat4 look_at(veekay::vec3 at) const;

    veekay::mat4 view_projection(float aspect_ratio, const veekay::mat4 &view) const;
};

inline struct {
    // Объекты для изображения, куда будет записываться информация о глубине
    VkFormat depth_image_format;
    VkImage depth_image;
    VkDeviceMemory depth_image_memory;
    VkImageView depth_image_view;

    VkShaderModule vertex_shader; // Простой шейдер для трансформации геометрии и не больше

    // Объекты графического конвейера и описания ресурсов шейдера для записи глубины в текстуру
    VkDescriptorSetLayout descriptor_set_layout;
    VkDescriptorSet descriptor_set;
    VkPipelineLayout pipeline_layout;
    VkPipeline pipeline;

    veekay::graphics::Buffer* uniform_buffer; // Буфер для единственной матрицы проекции теней
    VkSampler sampler; // Специальный сэмплер для текстуры (карты) теней

    veekay::mat4 matrix; // Сама матрица проекции теней
} shadow;

inline struct {
    VkImage depth_image;
    VkImageView depth_image_view;
    VkDeviceMemory depth_image_memory;
    VkFormat depth_image_format;

    VkDescriptorSet descriptor_set;

    veekay::graphics::Buffer* uniform_buffer;

    veekay::mat4 matrix;
} spotShadow;


#endif //VEEKAY_OBJECTS_H