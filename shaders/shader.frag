#version 450

struct PointLight {
	vec4 position_radius;
	vec3 color;
};

struct SpotLight {
	vec4 position_radius;
	vec4 color_angle;
	vec4 direction_outer_angle;
};

layout (location = 0) in vec3 f_position;
layout (location = 1) in vec3 f_normal;
layout (location = 2) in vec2 f_uv;

layout (location = 0) out vec4 final_color;

layout (binding = 0, std140) uniform SceneUniforms {
	mat4 view_projection;
	vec3 view_position;

	vec3 ambient_light_intensity;

	vec3 sun_light_direction;
	vec3 sun_light_color;

	layout (offset = 128) uint point_lights_count;
	layout (offset = 132) uint spot_lights_count;
};

layout (binding = 1, std140) uniform ModelUniforms {
	mat4 model;
	vec3 albedo_color;
	vec4 specular_color_shininess;
};

layout (binding = 2, std430) readonly buffer PointLights {
	PointLight point_lights[];
};

layout (binding = 3, std430) readonly buffer SpotLights {
	SpotLight spot_lights[];
};

void main() {
	float shininess = specular_color_shininess.w;

	vec3 specular_color = specular_color_shininess.xyz;

	vec3 normal = normalize(f_normal);

	vec3 norm_sun_light_dir = normalize(sun_light_direction);

	vec3 view_dir = normalize(view_position - f_position);
	vec3 half_vector = normalize(view_dir - norm_sun_light_dir);

	float sun_shade = max(0.0f, -dot(norm_sun_light_dir, normal));
	vec3 sun_diffuse = albedo_color;
	vec3 sun_specular = specular_color * pow(max(0.0f, dot(normal, half_vector)), shininess);

	vec3 sun_light_intensity = sun_shade * sun_light_color * (sun_diffuse + sun_specular);

	vec3 sun_color = ambient_light_intensity + sun_light_intensity;

	vec3 point_lights_color = vec3(0);

	float constant = 1;
	float linear = 0.09f;
	float quadratic = 0.05f;

	for (uint i = 0; i < point_lights_count; ++i) {
		PointLight light = point_lights[i];

		vec3 lpos = light.position_radius.xyz;
		float radius = light.position_radius.w;

		float dist = distance(f_position, lpos);

		float attenuation = 1.0f / (constant + linear * dist + quadratic * dist * dist);

		vec3 light_dir = normalize(lpos - f_position);

		half_vector = normalize(view_dir + light_dir);

		float shade = max(0.0f, dot(light_dir, normal));

		vec3 diffuse = albedo_color;
		vec3 specular = specular_color * pow(max(0.0f, dot(normal, half_vector)), shininess);

		vec3 light_intensity = attenuation * shade * (diffuse * light.color + specular);

		point_lights_color = vec3(attenuation, 0, 0);
//		point_lights_color += light_intensity;
	}

	vec3 spot_lights_color = vec3(0);

	for (uint i = 0; i < spot_lights_count; ++i) {
		SpotLight light = spot_lights[i];

		vec3 direction = light.direction_outer_angle.xyz;
		float outer_angle = light.direction_outer_angle.w;

		vec3 lpos = light.position_radius.xyz;
		float radius = light.position_radius.w;

		vec3 light_color = light.color_angle.xyz;
		float angle = light.color_angle.w;

		vec3 light_dir = normalize(lpos - f_position);

		if (distance(lpos, f_position) > radius) continue;

		float theta = dot(light_dir, normalize(-direction));

		if (theta > outer_angle) {
			half_vector = normalize(view_dir + light_dir);

			float shade = max(0.0f, dot(light_dir, normal));

			vec3 diffuse = albedo_color;
			vec3 specular = specular_color * pow(max(0.0f, dot(normal, half_vector)), shininess);

			float epsilon = angle - outer_angle;

			float dimming = clamp((theta - outer_angle) / epsilon, 0.0f, 1.0f);

			vec3 light_intensity = dimming * shade * light_color * (diffuse + specular);

			spot_lights_color += light_intensity;
		}
	}

	final_color = vec4(0 + point_lights_color + 0, 1.0f);
}
