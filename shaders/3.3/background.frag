#version 330
layout(location=0) out vec4 color;

in vec2 v_TexCoord;

uniform sampler2D u_Texture;
uniform int u_time;


void main(){
    color = vec4(0.2,0.3,0.2,1.0);
}