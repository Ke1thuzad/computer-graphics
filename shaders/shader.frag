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

vec3 blinn_phong(vec3 light_dir, vec3 light_color, vec3 view_dir, vec3 normal, vec3 albedo, vec3 specular_color, float shininess) {
	vec3 half_vector = normalize(light_dir + view_dir);

	float diffuse_strength = max(dot(normal, light_dir), 0.0f);
	vec3 diffuse = diffuse_strength * albedo;

	float specular_strength = pow(max(dot(normal, half_vector), 0.0f), shininess);
	vec3 specular = specular_strength * specular_color;

	return light_color * (diffuse + specular);
}

void main() {
	vec3 normal = normalize(f_normal);
	vec3 view_dir = normalize(view_position - f_position);

	float shininess = specular_color_shininess.w;
	vec3 specular_color = specular_color_shininess.xyz;

	vec3 norm_sun_light_dir = normalize(-sun_light_direction);
	vec3 sun_light_intensity = blinn_phong(
		norm_sun_light_dir,
		sun_light_color,
		view_dir,
		normal,
		albedo_color,
		specular_color,
		shininess
	);

	vec3 sun_color = ambient_light_intensity + sun_light_intensity;

	vec3 point_lights_color = vec3(0);
	float constant = 1.0f;
	float linear = 0.09f;
	float quadratic = 0.05f;

	for (uint i = 0; i < point_lights_count; ++i) {
		PointLight light = point_lights[i];

		vec3 lpos = light.position_radius.xyz;
		float radius = light.position_radius.w;

		float dist = distance(f_position, lpos);
		if (dist > radius) continue;

		float falloff = 1.0f - smoothstep(0.9f, 1.0f, dist / radius);
		float attenuation = falloff / (constant + linear * dist + quadratic * dist * dist);

		vec3 light_dir = normalize(lpos - f_position);

		vec3 light_intensity = blinn_phong(
		light_dir,
		light.color,
		view_dir,
		normal,
		albedo_color,
		specular_color,
		shininess
		);

		point_lights_color += attenuation * light_intensity;
	}

	// Прожекторы
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
		float dist = distance(lpos, f_position);

		if (dist > radius) continue;

		float theta = dot(light_dir, normalize(-direction));

		if (theta > outer_angle) {
			float epsilon = angle - outer_angle;
			float falloff = 1.0f - smoothstep(0.9f, 1.0f, dist / radius);
			float dimming = clamp((theta - outer_angle) / epsilon, 0.0f, 1.0f);

			vec3 light_intensity = blinn_phong(
			light_dir,
			light_color,
			view_dir,
			normal,
			albedo_color,
			specular_color,
			shininess
			);

			spot_lights_color += falloff * dimming * light_intensity;
		}
	}

	final_color = vec4(sun_color + point_lights_color + spot_lights_color, 1.0f);
}