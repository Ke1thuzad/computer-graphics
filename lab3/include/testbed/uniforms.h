#ifndef VEEKAY_UNIFORMS_H
#define VEEKAY_UNIFORMS_H


struct MaterialUniform {
    veekay::vec3 albedo_color;
    float _pad0;
    veekay::vec3 specular_color;
    float shininess;
};

struct SceneUniforms {
    veekay::mat4 view_projection;
    veekay::vec3 view_position;
    float _pad0;

    veekay::vec3 ambient_light_intensity;
    float _pad1;

    veekay::vec3 sun_light_direction;
    float _pad2;
    veekay::vec3 sun_light_color;
    float _pad3;

    uint32_t point_lights_count;
    uint32_t spot_lights_count;
};

struct ModelUniforms {
    veekay::mat4 model;

    MaterialUniform material;
};

#endif //VEEKAY_UNIFORMS_H