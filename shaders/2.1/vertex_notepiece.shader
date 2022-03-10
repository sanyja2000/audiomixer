attribute vec3 position;
attribute vec2 texCoord;
attribute vec3 normal;
varying vec2 v_TexCoord;
varying vec3 v_position;
varying vec3 v_normal;

uniform mat4 u_MVP;
uniform float u_Time;
uniform float u_LastPlayed;

void main()
{
    gl_Position = u_MVP * vec4(position.x,position.y-0.3*sin((u_Time-u_LastPlayed)*3.0)*pow(1.0/(u_Time-u_LastPlayed+0.1),2.0),position.z,1.0);
    v_TexCoord = texCoord;
    v_normal = normalize(normal).xyz;
    v_position = position.xyz;
}
