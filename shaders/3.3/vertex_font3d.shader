#version 330
layout(location=0) in vec2 position;
layout(location=1) in vec2 texCoord;
layout(location=2) in float charIndex;
out vec2 v_TexCoord;
out float v_charIndex;

uniform vec4 u_PosRotScl;
uniform mat4 u_MVP;
uniform int u_is3d;

void main()
{
    if(u_is3d==0){
        gl_Position = vec4(position.x*u_PosRotScl.w+u_PosRotScl.x,position.y*u_PosRotScl.w+u_PosRotScl.y,-1.0,1.0); // check this z
    }
    else{
        gl_Position = u_MVP * vec4(vec3(position,0.0),1.0);
    }
    v_TexCoord = texCoord;
    v_charIndex = charIndex;
}