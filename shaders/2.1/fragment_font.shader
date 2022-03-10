varying vec2 v_TexCoord;
varying float v_charIndex;
uniform sampler2D u_Texture;

void main(){
    float x = 1.0/16.0*mod(int(v_charIndex),16);
    float y = 1.0/16.0*(15.0-floor(v_charIndex/16));
    vec4 texColor = texture(u_Texture, vec2(x,y)+v_TexCoord);
    if(texColor.r<0.3){gl_FragColor = vec4(0.0,0.0,0.0,0.0);}
    else{gl_FragColor = texColor;}
    //color = vec4(0.8,0.3,0.2,1.0);
}