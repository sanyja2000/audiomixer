#version 330 core
layout(location=0) in vec2 position;
layout(location=1) in vec2 texCoord;
out vec2 v_TexCoord;

void main()
{
    // Outputs the input vertex position
    gl_Position = vec4(position.x,position.y,0.99,1.0);

    // Outputs UV-coordinates
    v_TexCoord = texCoord.xy;
}
