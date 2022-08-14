#version 330
layout(location=0) out vec4 color;

in vec2 v_TexCoord;
uniform sampler2D u_Texture;
uniform float ymax;
uniform float scrollOffset;
void main(){
    
    vec2 newTc = vec2(v_TexCoord.x,(v_TexCoord.y-(1-ymax))/ymax);

    if(newTc.y<0){
        color = vec4(0.0,0.0,0.0,0.6);
    }else{
        newTc.y = mod(newTc.y-scrollOffset,1);
        vec4 texColor = texture(u_Texture, newTc);
    
    
        if(texColor.a < 0.2){
            color = vec4(0.0,0.0,0.0,0.6);
        }
        else{
            
            color = texColor;
        }
    }
    
    //;


    /*
    if(v_TexCoord.x<0.01){
        color = vec4(0.3,0.3,0.3,0.6);
    }*/
}