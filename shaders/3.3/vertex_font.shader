#version 330
layout(location=0) in vec2 position;
layout(location=1) in vec2 texCoord;
layout(location=2) in float charIndex;
out vec2 v_TexCoord;
out float v_charIndex;

uniform vec4 u_PosRotScl;
void main()
{
    gl_Position = vec4(position.x*u_PosRotScl.w+u_PosRotScl.x,position.y*u_PosRotScl.w+u_PosRotScl.y,-1.0,1.0); // check this z
    v_TexCoord = texCoord;
    v_charIndex = charIndex;
}