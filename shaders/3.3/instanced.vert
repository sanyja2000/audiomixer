#version 330 core
layout(location=0) in vec3 position;
layout(location=1) in vec2 texCoord;
layout(location=2) in vec3 normal;
layout(location=3) in vec3 offset;
varying vec2 v_TexCoord;
varying vec3 v_position;
varying vec3 v_normal;
varying vec3 lightPos;

uniform mat4 u_MVP;

void main()
{

    gl_Position = u_MVP * vec4(position,1.0)+vec4(offset,0.0);
    //gl_Position = vec4(position*0.03,1.0)+vec4(offset/0.03,1.0);
    
    lightPos = (u_MVP * vec4(0.0,4.0,0.0,1.0)).xyz;
    v_TexCoord = texCoord;
    v_normal = normalize(u_MVP * vec4(normal,0.0)).xyz;
    v_position = (u_MVP * vec4(position.xyz,1.0)).xyz;
    
}
