attribute vec2 position;
attribute vec2 texCoord;
varying vec2 v_TexCoord;

uniform float xcoord;
uniform vec3 xyzoom;

void main()
{
    gl_Position = vec4(position.x+xcoord,position.y,0.2,1.0);
    v_TexCoord = texCoord.xy;
    //v_position = position.xy;
}
