#version 330
layout(location=0) out vec4 color;

in vec2 v_TexCoord;
uniform sampler2D u_Texture;
uniform int u_time;
void main(){

    
    color = vec4(0.0,0.0,0.0,0.6);
    if(v_TexCoord.x<0.01){
        color = vec4(0.3,0.3,0.3,0.6);
    }
}