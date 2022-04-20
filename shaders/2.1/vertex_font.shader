
attribute vec2 position;
attribute vec2 texCoord;
attribute float charIndex;
varying vec2 v_TexCoord;
varying float v_charIndex;

uniform vec4 u_PosRotScl;
void main()
{
    //gl_Position = vec4(position.x*u_PosRotScl.w+u_PosRotScl.x,position.y*u_PosRotScl.w+u_PosRotScl.y,-1.0,1.0); // check this z
    if(u_is3d==0){
        gl_Position = vec4(position.x*u_PosRotScl.w+u_PosRotScl.x,position.y*u_PosRotScl.w+u_PosRotScl.y,-1.0,1.0); // check this z
    }
    else{
        gl_Position = u_MVP * vec4(vec3(position,0.0),1.0);
    }
    v_TexCoord = texCoord;
    v_charIndex = charIndex;
}