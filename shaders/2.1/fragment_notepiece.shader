varying vec2 v_TexCoord;
varying vec3 v_position;
varying vec3 v_normal;

uniform sampler2D u_Texture;
uniform float u_Time;
uniform float u_LastPlayed;

void main(){
    vec4 texColor = vec4(0.0,0.0,0.0,1.0);//texture(u_Texture, v_TexCoord);
    
    float sum = v_position.x + v_position.y + v_position.z;
    if(mod(sum,1.0)<0.1){
      texColor+=vec4(0.5,0.5,0.5,0.0);
    }

    texColor.x *= (1.0/(u_Time-u_LastPlayed));
    texColor.y *= (1.0/(u_Time-u_LastPlayed));
    vec3 lightPos = vec3(0.0,4.0,3.0);

     vec3 lightDir = normalize(lightPos-v_position);
      vec3 norm = v_normal;
     
      float diff = max(dot(norm, lightDir), 0.0);
    
    gl_FragColor = vec4(texColor.xyz*(0.3+diff),1.0);
    //color = vec4(0.8,0.3,0.2,1.0);
}
