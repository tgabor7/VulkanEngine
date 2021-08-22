#version 450

layout(location = 0) in vec3 fragColor;

layout (location = 0) out vec4 FragColor;

layout(push_constant) uniform Push {
	mat4 transform; // projection * view * model
	mat4 normalMatrix;
} push;

void main(){
	FragColor = vec4(fragColor, 1.0);
}