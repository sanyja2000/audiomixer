#version 330 core
layout(location=0) in vec2 position;
layout(location=1) in vec2 texCoord;
out vec2 v_TexCoord;
//varying vec2 v_position;

uniform float xcoord;

void main()
{
    
    gl_Position = vec4(position.x+xcoord,position.y,-0.1,1.0);
    v_TexCoord = texCoord.xy;
    //v_position = position.xy;
}
