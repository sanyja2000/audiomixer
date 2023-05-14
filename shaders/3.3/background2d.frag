#version 330
layout(location=0) out vec4 color;

in vec2 v_TexCoord;

uniform sampler2D u_Texture;
uniform int u_time;

const float scaler = log(100)/log(10);

void main(){

    color = vec4(0.05,0.05,0.05,1.0);
    float uvx = v_TexCoord.x;
    vec4 texcolor = texture(u_Texture,vec2(uvx,v_TexCoord.y));
    float n = (texcolor.g+texcolor.r*256)/1000;

    if(v_TexCoord.y-0.5<n && v_TexCoord.y>0.5){
        color = vec4(v_TexCoord.x,0.5,0,1.0);
    }
    
    if(v_TexCoord.y>0.5-n && v_TexCoord.y<0.5){
        color = vec4(v_TexCoord.x,0.6,0,1.0);
    }
    
}