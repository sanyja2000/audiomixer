#version 330
layout(location=0) out vec4 color;

in vec2 v_TexCoord;

uniform sampler2D u_Texture;
uniform int u_time;
/*
uniform float selectedIndex;
uniform float ymax;
uniform float scrollOffset;
*/
const float scaler = log(100)/log(10);

void main(){

    //float RES = 3072/8+1;

    color = vec4(0.05,0.05,0.05,1.0);

    //float uvx = pow(10, v_TexCoord.x*scaler)/100;
    float uvx = v_TexCoord.x;
    vec4 texcolor = texture(u_Texture,vec2(uvx,v_TexCoord.y));
    float n = (texcolor.g+texcolor.r*256)/1000;//+(texcolor.a+texcolor.b*256)/1000;

    /*
    if(v_TexCoord.x<0.1 || v_TexCoord.x > 0.9){
        return;
    }
    vec4 texColor = texture(u_Texture,vec2(v_TexCoord.x*3/10-0.1,0.0));
    */
    /*
    vec4 texColor = texture(u_Texture,vec2(v_TexCoord.x,0.0));

    float n = pow((texColor.r+texColor.g)/2,10)*0.5;
    */
    if(v_TexCoord.y-0.5<n && v_TexCoord.y>0.5){
        color = vec4(v_TexCoord.x,0.5,0,1.0);
    }
    
    if(v_TexCoord.y>0.5-n && v_TexCoord.y<0.5){
        color = vec4(v_TexCoord.x,0.6,0,1.0);
    }
    
    


}