#define PI 3.1415926535
varying vec2 v_TexCoord;
varying vec3 v_position;
varying vec3 v_normal;

uniform sampler2D u_Texture;
uniform float u_Time;

float rand(vec2 c){
	return fract(sin(dot(c.xy ,vec2(12.9898,78.233))) * 43758.5453);
}

float noise(vec2 p, float freq ){
	float unit = 1024.0/freq;
	vec2 ij = floor(p/unit);
	vec2 xy = mod(p,unit)/unit;
	//xy = 3.*xy*xy-2.*xy*xy*xy;
	xy = .5*(1.-cos(PI*xy));
	float a = rand((ij+vec2(0.,0.)));
	float b = rand((ij+vec2(1.,0.)));
	float c = rand((ij+vec2(0.,1.)));
	float d = rand((ij+vec2(1.,1.)));
	float x1 = mix(a, b, xy.x);
	float x2 = mix(c, d, xy.x);
	return mix(x1, x2, xy.y);
}

float pNoise(vec2 p, int res){
	float persistance = .5;
	float n = 0.;
	float normK = 0.;
	float f = 4.;
	float amp = 1.;
	int iCount = 0;
	for (int i = 0; i<50; i++){
		n+=amp*noise(p, f);
		f*=2.;
		normK+=amp;
		amp*=persistance;
		if (iCount == res) break;
		iCount++;
	}
	float nf = n/normK;
	return nf*nf*nf*nf;
}


void main(){

    float noiseVal = pNoise(v_TexCoord*10000.0+vec2(u_Time*100,u_Time/2*100), 3)+0.1;

    vec3 col = vec3(1.0,1.0,1.0);

    if(noiseVal<0.12+sin(u_Time/3)*cos(u_Time/3)/100){
        noiseVal = 1.0;
        col = vec3(1.0,0.2,1.0);
    }

    gl_FragColor = vec4(col*noiseVal*0.5,1.0);
    //color = vec4(0.8,0.3,0.2,1.0);
}
