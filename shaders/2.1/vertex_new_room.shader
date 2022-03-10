attribute vec3 position;
attribute vec2 texCoord;
attribute vec3 normal;
varying vec2 v_TexCoord;
varying vec3 v_position;
varying vec3 v_normal;

uniform mat4 u_MVP;
void main()
{

    gl_Position = u_MVP * vec4(position,1.0);
    v_TexCoord = texCoord;
    v_normal = normalize(normal).xyz;
    v_position = position.xyz;
    
}
