
const RFDETR = (() => {
const getTensorBuffer = (safetensorBuffer, tensorMetadata) => {
  return safetensorBuffer.subarray(...tensorMetadata.data_offsets);
};

const getTensorMetadata = (safetensorBuffer) => {
    const metadataLength = Number(new DataView(safetensorBuffer.buffer).getBigUint64(0, true));
    const metadata = JSON.parse(new TextDecoder("utf8").decode(safetensorBuffer.subarray(8, 8 + metadataLength)));
    return Object.fromEntries(Object.entries(metadata).filter(([k, v]) => k !== "__metadata__").map(([k, v]) => [k, {...v, data_offsets: v.data_offsets.map(x => 8 + metadataLength + x)}]));
};

const createEmptyBuf = (device, size) => {
    return device.createBuffer({size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
};

const createUniformBuf = (device, size) => {
  return device.createBuffer({size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST})
}

const createInfinityUniformBuf = (device) => {
  const size = 4;
  const buf = device.createBuffer({
    mappedAtCreation: true,
    size,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
  });
  new Float32Array(buf.getMappedRange())[0] = Infinity;
  buf.unmap();
  return buf;
};

const createWeightBuf = (device, size, data) => {
  const buf = device.createBuffer({ size, usage: GPUBufferUsage.STORAGE, mappedAtCreation: true });
  new Uint8Array(buf.getMappedRange()).set(data); buf.unmap();
  return buf;
};

const addComputePass = (device, commandEncoder, pipeline, layout, infinityUniformBuf, bufs, workgroup) => {
  const bindGroup = device.createBindGroup({
    layout: layout,
    entries: [
      { binding: 0, resource: { buffer: infinityUniformBuf } },
      ...bufs.map((buffer, index) => ({ binding: index + 1, resource: { buffer } }))
    ]
  });

  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(...workgroup);
  passEncoder.end();
};

const E_4608_32_3 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_442368:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_442368:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 4608 */
  var lidx0 = i32(lindex.x); /* 32 */
  var alu0 = ((gidx0*96)+(lidx0*3));
  var alu1 = (alu0+1);
  var val0 = data1_442368[alu1];
  var alu2 = (alu0+2);
  var val1 = data1_442368[alu2];
  var val2 = data1_442368[alu0];
  data0_442368[alu1] = (val0*0.00392156862745098f);
  data0_442368[alu2] = (val2*0.00392156862745098f);
  data0_442368[alu0] = (val1*0.00392156862745098f);
}`;

const r_30_8_16_5_2_2_256 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_76800:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_998400:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_196608:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_768:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,20>;
  var gidx0 = i32(gindex.x); /* 8 */
  var gidx1 = i32(gindex.y); /* 30 */
  var lidx0 = i32(lindex.x); /* 16 */
  var cast0 = bitcast<u32>(gidx0);
  var alu0 = (gidx1*2560);
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  acc0[4] = 0.0f;
  acc0[5] = 0.0f;
  acc0[6] = 0.0f;
  acc0[7] = 0.0f;
  acc0[8] = 0.0f;
  acc0[9] = 0.0f;
  acc0[10] = 0.0f;
  acc0[11] = 0.0f;
  acc0[12] = 0.0f;
  acc0[13] = 0.0f;
  acc0[14] = 0.0f;
  acc0[15] = 0.0f;
  acc0[16] = 0.0f;
  acc0[17] = 0.0f;
  acc0[18] = 0.0f;
  acc0[19] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 256; Ridx0++) {
    var alu21 = (alu0+Ridx0);
    var val0 = data1_998400[alu21];
    var alu22 = (bitcast<i32>((cast0<<13u))+bitcast<i32>((bitcast<u32>(lidx0)<<8u))+Ridx0);
    var val1 = data2_196608[(alu22+131072)];
    var val2 = data1_998400[(alu21+1280)];
    var val3 = data2_196608[(alu22+135168)];
    var val4 = data1_998400[(alu21+256)];
    var val5 = data1_998400[(alu21+512)];
    var val6 = data1_998400[(alu21+1792)];
    var val7 = data1_998400[(alu21+768)];
    var val8 = data1_998400[(alu21+1536)];
    var val9 = data1_998400[(alu21+2048)];
    var val10 = data1_998400[(alu21+1024)];
    var val11 = data1_998400[(alu21+2304)];
    acc0[0] = (acc0[0]+(val0*val1));
    acc0[1] = (acc0[1]+(val2*val1));
    acc0[2] = (acc0[2]+(val0*val3));
    acc0[3] = (acc0[3]+(val2*val3));
    acc0[4] = (acc0[4]+(val4*val1));
    acc0[5] = (acc0[5]+(val8*val1));
    acc0[6] = (acc0[6]+(val4*val3));
    acc0[7] = (acc0[7]+(val8*val3));
    acc0[8] = (acc0[8]+(val5*val1));
    acc0[9] = (acc0[9]+(val6*val1));
    acc0[10] = (acc0[10]+(val5*val3));
    acc0[11] = (acc0[11]+(val6*val3));
    acc0[12] = (acc0[12]+(val7*val1));
    acc0[13] = (acc0[13]+(val9*val1));
    acc0[14] = (acc0[14]+(val7*val3));
    acc0[15] = (acc0[15]+(val9*val3));
    acc0[16] = (acc0[16]+(val10*val1));
    acc0[17] = (acc0[17]+(val11*val1));
    acc0[18] = (acc0[18]+(val10*val3));
    acc0[19] = (acc0[19]+(val11*val3));
  }
  var alu44 = (lidx0+bitcast<i32>((cast0<<5u)));
  var val12 = data3_768[(alu44+512)];
  var val13 = data3_768[(alu44+528)];
  var alu45 = (alu44+alu0);
  data0_76800[alu45] = (acc0[0]+val12);
  data0_76800[(alu45+16)] = (acc0[2]+val13);
  data0_76800[(alu45+256)] = (acc0[4]+val12);
  data0_76800[(alu45+272)] = (acc0[6]+val13);
  data0_76800[(alu45+512)] = (acc0[8]+val12);
  data0_76800[(alu45+528)] = (acc0[10]+val13);
  data0_76800[(alu45+768)] = (acc0[12]+val12);
  data0_76800[(alu45+784)] = (acc0[14]+val13);
  data0_76800[(alu45+1024)] = (acc0[16]+val12);
  data0_76800[(alu45+1040)] = (acc0[18]+val13);
  data0_76800[(alu45+1280)] = (acc0[1]+val12);
  data0_76800[(alu45+1296)] = (acc0[3]+val13);
  data0_76800[(alu45+1536)] = (acc0[5]+val12);
  data0_76800[(alu45+1552)] = (acc0[7]+val13);
  data0_76800[(alu45+1792)] = (acc0[9]+val12);
  data0_76800[(alu45+1808)] = (acc0[11]+val13);
  data0_76800[(alu45+2048)] = (acc0[13]+val12);
  data0_76800[(alu45+2064)] = (acc0[15]+val13);
  data0_76800[(alu45+2304)] = (acc0[17]+val12);
  data0_76800[(alu45+2320)] = (acc0[19]+val13);
}`;

const E_9216_3_16 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_442368:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_442368:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 3 */
  var gidx1 = i32(gindex.y); /* 9216 */
  var lidx0 = i32(lindex.x); /* 16 */
  var alu0 = ((gidx1*48)+(lidx0*3));
  var val0 = data1_442368[(gidx0+alu0)];
  data0_442368[((alu0-gidx0)+2)] = val0;
}`;

const E_192_24_16_3_2 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_442368:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_442368:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 24 */
  var gidx1 = i32(gindex.y); /* 192 */
  var lidx0 = i32(lindex.x); /* 16 */
  var alu0 = (lidx0+bitcast<i32>((bitcast<u32>(gidx0)<<4u)));
  var alu1 = ((f32((alu0+1)))+-1.0f);
  var alu2 = select(alu1,383.0f,(383.0f<alu1));
  var alu3 = trunc(alu2);
  var cast0 = (i32(alu3));
  var alu4 = (gidx1*2304);
  var alu5 = select(cast0,(i32((alu3+1.0f))),(alu3<alu2));
  var alu6 = (alu4+(alu5*3));
  var alu7 = ((-1<alu5)&(alu5<384));
  var val0 = select(0.0f, data1_442368[(alu6+1)], alu7);
  var val1 = select(0.0f, data1_442368[(alu6+2)], alu7);
  var val2 = select(0.0f, data1_442368[(alu6+1152)], alu7);
  var val3 = select(0.0f, data1_442368[(alu6+1153)], alu7);
  var val4 = select(0.0f, data1_442368[(alu6+1154)], alu7);
  var alu8 = (alu3+-1.0f);
  var alu9 = (alu2<alu3);
  var alu10 = select(cast0,(i32(alu8)),alu9);
  var alu11 = (alu4+(alu10*3));
  var alu12 = ((-1<alu10)&(alu10<384));
  var val5 = select(0.0f, data1_442368[(alu11+1)], alu12);
  var val6 = select(0.0f, data1_442368[(alu11+2)], alu12);
  var val7 = select(0.0f, data1_442368[(alu11+1152)], alu12);
  var val8 = select(0.0f, data1_442368[(alu11+1153)], alu12);
  var val9 = select(0.0f, data1_442368[(alu11+1154)], alu12);
  var val10 = select(0.0f, data1_442368[alu6], alu7);
  var val11 = select(0.0f, data1_442368[alu11], alu12);
  var alu13 = (alu0+(gidx1*768));
  var alu14 = select(alu3,alu8,alu9);
  var alu15 = (alu2-alu14);
  data0_442368[alu13] = (val6+((val1-val6)*alu15));
  data0_442368[(alu13+384)] = (val9+((val4-val9)*alu15));
  data0_442368[(alu13+147456)] = (val5+((val0-val5)*alu15));
  data0_442368[(alu13+147840)] = (val8+((val3-val8)*alu15));
  data0_442368[(alu13+294912)] = (val11+((val10-val11)*alu15));
  data0_442368[(alu13+295296)] = (val7+((val2-val7)*alu15));
}`;

const E_384_12_16_3_2 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_442368:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_442368:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_3:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_3:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 12 */
  var gidx1 = i32(gindex.y); /* 384 */
  var lidx0 = i32(lindex.x); /* 16 */
  var alu0 = ((f32((gidx1+1)))+-1.0f);
  var alu1 = select(alu0,383.0f,(383.0f<alu0));
  var alu2 = trunc(alu1);
  var cast0 = (i32(alu2));
  var alu3 = (lidx0+bitcast<i32>((bitcast<u32>(gidx0)<<5u)));
  var alu4 = select(cast0,(i32((alu2+1.0f))),(alu2<alu1));
  var alu5 = (alu3+(alu4*384));
  var alu6 = ((-1<alu4)&(alu4<384));
  var val0 = select(0.0f, data1_442368[alu5], alu6);
  var alu7 = (alu2+-1.0f);
  var alu8 = (alu1<alu2);
  var alu9 = select(cast0,(i32(alu7)),alu8);
  var alu10 = (alu3+(alu9*384));
  var alu11 = ((-1<alu9)&(alu9<384));
  var val1 = select(0.0f, data1_442368[alu10], alu11);
  var val2 = data2_3[0];
  var val3 = data2_3[1];
  var val4 = data3_3[0];
  var val5 = select(0.0f, data1_442368[(alu5+147456)], alu6);
  var val6 = select(0.0f, data1_442368[(alu10+147456)], alu11);
  var val7 = data3_3[1];
  var val8 = select(0.0f, data1_442368[(alu5+147472)], alu6);
  var val9 = select(0.0f, data1_442368[(alu5+294912)], alu6);
  var val10 = select(0.0f, data1_442368[(alu10+294912)], alu11);
  var val11 = data2_3[2];
  var val12 = data3_3[2];
  var val13 = select(0.0f, data1_442368[(alu5+16)], alu6);
  var val14 = select(0.0f, data1_442368[(alu5+294928)], alu6);
  var val15 = select(0.0f, data1_442368[(alu10+16)], alu11);
  var val16 = select(0.0f, data1_442368[(alu10+147472)], alu11);
  var val17 = select(0.0f, data1_442368[(alu10+294928)], alu11);
  var alu12 = ((gidx0*96)+(lidx0*3)+(gidx1*1152));
  var alu13 = (1/val4);
  var alu14 = (1/val7);
  var alu15 = (1/val12);
  var alu16 = select(alu2,alu7,alu8);
  var alu17 = (alu1-alu16);
  data0_442368[(alu12+48)] = (((val15+((val13-val15)*alu17))-val2)*alu13);
  data0_442368[(alu12+49)] = (((val16+((val8-val16)*alu17))-val3)*alu14);
  data0_442368[(alu12+50)] = (((val17+((val14-val17)*alu17))-val11)*alu15);
  data0_442368[(alu12+1)] = (((val6+((val5-val6)*alu17))-val3)*alu14);
  data0_442368[(alu12+2)] = (((val10+((val9-val10)*alu17))-val11)*alu15);
  data0_442368[alu12] = (((val1+((val0-val1)*alu17))-val2)*alu13);
}`;

const r_577_48_16_2_4_16_3 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32,128>;
@group(0) @binding(1)var<storage,read_write>data0_221568:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_384:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_442368:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_294912:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_384:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_221568:array<f32>;
@compute @workgroup_size(16,2) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,4>;
  var acc1: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 48 */
  var gidx1 = i32(gindex.y); /* 577 */
  var lidx0 = i32(lindex.x); /* 16 */
  var lidx1 = i32(lindex.y); /* 2 */
  var alu0 = (gidx1+23);
  var alu1 = ((alu0*683)>>14u);
  var alu2 = (0<gidx1);
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  for (var Ridx1 = 0; Ridx1 < 16; Ridx1++) {
    var alu7 = ((lidx0*3)+((alu0-(24*alu1))*48)+(alu1*18432)+(Ridx1*1152));
    var val0 = select(0.0f, data2_442368[(alu7+-18432)], alu2);
    var alu8 = (lidx0+bitcast<i32>((bitcast<u32>(Ridx1)<<4u))+(gidx0*6144)+(lidx1*3072));
    var val1 = data3_294912[alu8];
    var val2 = select(0.0f, data2_442368[(alu7+-18431)], alu2);
    var val3 = data3_294912[(alu8+256)];
    var val4 = select(0.0f, data2_442368[(alu7+-18430)], alu2);
    var val5 = data3_294912[(alu8+512)];
    var val6 = data3_294912[(alu8+768)];
    var val7 = data3_294912[(alu8+1024)];
    var val8 = data3_294912[(alu8+1280)];
    var val9 = data3_294912[(alu8+1536)];
    var val10 = data3_294912[(alu8+1792)];
    var val11 = data3_294912[(alu8+2048)];
    var val12 = data3_294912[(alu8+2304)];
    var val13 = data3_294912[(alu8+2560)];
    var val14 = data3_294912[(alu8+2816)];
    acc0[0] = (acc0[0]+(val0*val1)+(val2*val3)+(val4*val5));
    acc0[1] = (acc0[1]+(val0*val6)+(val2*val7)+(val4*val8));
    acc0[2] = (acc0[2]+(val0*val9)+(val2*val10)+(val4*val11));
    acc0[3] = (acc0[3]+(val0*val12)+(val2*val13)+(val4*val14));
  }
  var cast0 = bitcast<u32>(lidx1);
  var cast1 = bitcast<i32>((cast0<<6u));
  var alu14 = (bitcast<i32>((bitcast<u32>(lidx0)<<2u))+cast1);
  temp0[alu14] = acc0[0];
  temp0[(alu14+1)] = acc0[1];
  temp0[(alu14+2)] = acc0[2];
  temp0[(alu14+3)] = acc0[3];
  workgroupBarrier();
  acc1[0] = 0.0f;
  acc1[1] = 0.0f;
  acc1[2] = 0.0f;
  acc1[3] = 0.0f;
  for (var Ridx105 = 0; Ridx105 < 16; Ridx105++) {
    var alu24 = (cast1+bitcast<i32>((bitcast<u32>(Ridx105)<<2u)));
    var val15 = temp0[alu24];
    var val16 = temp0[(alu24+1)];
    var val17 = temp0[(alu24+2)];
    var val18 = temp0[(alu24+3)];
    acc1[0] = (acc1[0]+val15);
    acc1[1] = (acc1[1]+val16);
    acc1[2] = (acc1[2]+val17);
    acc1[3] = (acc1[3]+val18);
  }
  var alu30 = (bitcast<i32>((bitcast<u32>(gidx0)<<3u))+bitcast<i32>((cast0<<2u)));
  var val19 = data1_384[alu30];
  var val20 = data4_384[alu30];
  var alu31 = (alu30+(gidx1*384));
  var val21 = data5_221568[alu31];
  var alu32 = (alu30+1);
  var val22 = data1_384[alu32];
  var val23 = data4_384[alu32];
  var alu33 = (alu31+1);
  var val24 = data5_221568[alu33];
  var alu34 = (alu30+2);
  var val25 = data1_384[alu34];
  var alu35 = (alu30+3);
  var val26 = data1_384[alu35];
  var val27 = data4_384[alu34];
  var alu36 = (alu31+2);
  var val28 = data5_221568[alu36];
  var val29 = data4_384[alu35];
  var alu37 = (alu31+3);
  var val30 = data5_221568[alu37];
  var alu38 = (lidx0==0);
  var alu39 = (gidx1<1);
  var alu40 = select(0.0f,val19,alu39);
  var alu41 = select((acc1[0]+val20),0.0f,alu39);
  var alu42 = select(0.0f,val22,alu39);
  var alu43 = select((acc1[1]+val23),0.0f,alu39);
  var alu44 = select(0.0f,val25,alu39);
  var alu45 = select((acc1[2]+val27),0.0f,alu39);
  var alu46 = select(0.0f,val26,alu39);
  var alu47 = select((acc1[3]+val29),0.0f,alu39);
  if (alu38) {
    data0_221568[alu31] = (alu40+alu41+val21);
  }
  if (alu38) {
    data0_221568[alu33] = (alu42+alu43+val24);
  }
  if (alu38) {
    data0_221568[alu36] = (alu44+alu45+val28);
  }
  if (alu38) {
    data0_221568[alu37] = (alu46+alu47+val30);
  }
}`;

const r_2_2_145_32_12 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32,32>;
@group(0) @binding(1)var<storage,read_write>data0_580:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_221568:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,1>;
  var acc1: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 145 */
  var gidx1 = i32(gindex.y); /* 2 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 32 */
  var alu0 = (gidx0+11);
  var alu1 = ((alu0*171)>>11u);
  acc0[0] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 12; Ridx0++) {
    var alu3 = ((lidx0*12)+Ridx0);
    var val0 = select(0.0f, data1_221568[((gidx1*4608)+((alu0-(12*alu1))*384)+(alu1*9216)+(gidx2*110592)+alu3+-8832)], (0<gidx0));
    var val1 = data1_221568[alu3];
    var alu4 = select(0.0f,val1,(gidx0<1));
    acc0[0] = (acc0[0]+alu4+val0);
  }
  temp0[lidx0] = acc0[0];
  workgroupBarrier();
  acc1[0] = 0.0f;
  for (var Ridx103 = 0; Ridx103 < 32; Ridx103++) {
    var val2 = temp0[Ridx103];
    acc1[0] = (acc1[0]+val2);
  }
  var alu12 = (lidx0==0);
  if (alu12) {
    data0_580[(gidx0+(gidx1*145)+(gidx2*290))] = (acc1[0]*0.0026041666666666665f);
  }
}`;

const r_2_2_145_32_12n1 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32,32>;
@group(0) @binding(1)var<storage,read_write>data0_580:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_221568:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_580:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,1>;
  var acc1: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 145 */
  var gidx1 = i32(gindex.y); /* 2 */
  var gidx2 = i32(gindex.z); /* 2 */
  var alu0 = (gidx0+(gidx1*145)+(gidx2*290));
  var val0 = data2_580[alu0];
  var lidx0 = i32(lindex.x); /* 32 */
  var alu1 = (gidx0+11);
  var alu2 = ((alu1*171)>>11u);
  acc0[0] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 12; Ridx0++) {
    var alu4 = ((lidx0*12)+Ridx0);
    var val1 = select(0.0f, data1_221568[((gidx1*4608)+((alu1-(12*alu2))*384)+(alu2*9216)+(gidx2*110592)+alu4+-8832)], (0<gidx0));
    var val2 = data1_221568[alu4];
    var alu5 = select(0.0f,val2,(gidx0<1));
    var alu6 = ((alu5+val1)-val0);
    acc0[0] = (acc0[0]+(alu6*alu6));
  }
  temp0[lidx0] = acc0[0];
  workgroupBarrier();
  acc1[0] = 0.0f;
  for (var Ridx103 = 0; Ridx103 < 32; Ridx103++) {
    var val3 = temp0[Ridx103];
    acc1[0] = (acc1[0]+val3);
  }
  var alu14 = (lidx0==0);
  if (alu14) {
    data0_580[alu0] = (1/sqrt(((acc1[0]*0.0026041666666666665f)+1e-05f)));
  }
}`;

const E_2_145_24_16_2 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_222720:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_221568:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_580:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_580:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_384:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_384:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 24 */
  var lidx0 = i32(lindex.x); /* 16 */
  var alu0 = (lidx0+bitcast<i32>((bitcast<u32>(gidx0)<<4u)));
  var val0 = data1_221568[alu0];
  var gidx1 = i32(gindex.y); /* 145 */
  var gidx2 = i32(gindex.z); /* 2 */
  var alu1 = (gidx1+11);
  var alu2 = ((alu1*171)>>11u);
  var alu3 = (alu0+(alu2*9216)+((alu1-(12*alu2))*384)+(gidx2*110592));
  var alu4 = (0<gidx1);
  var val1 = select(0.0f, data1_221568[(alu3+-8832)], alu4);
  var alu5 = (gidx1+(gidx2*290));
  var val2 = data2_580[alu5];
  var val3 = data3_580[alu5];
  var val4 = data4_384[alu0];
  var val5 = data5_384[alu0];
  var val6 = select(0.0f, data1_221568[(alu3+-4224)], alu4);
  var alu6 = (alu5+145);
  var val7 = data2_580[alu6];
  var val8 = data3_580[alu6];
  var alu7 = (alu0+(gidx1*384)+(gidx2*111360));
  var alu8 = select(0.0f,val0,(gidx1<1));
  data0_222720[alu7] = ((((alu8+val1)-val2)*val3*val4)+val5);
  data0_222720[(alu7+55680)] = ((((alu8+val6)-val7)*val8*val4)+val5);
}`;

const r_20_48_29_4_2_384 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_222720:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_222720:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_147456:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_384:array<f32>;
@compute @workgroup_size(29) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,8>;
  var gidx0 = i32(gindex.x); /* 48 */
  var gidx1 = i32(gindex.y); /* 20 */
  var lidx0 = i32(lindex.x); /* 29 */
  var alu0 = ((gidx1*11136)+(lidx0*384));
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  acc0[4] = 0.0f;
  acc0[5] = 0.0f;
  acc0[6] = 0.0f;
  acc0[7] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 384; Ridx0++) {
    var val0 = data1_222720[(alu0+Ridx0)];
    var alu9 = ((gidx0*3072)+Ridx0);
    var val1 = data2_147456[(alu9+2304)];
    var val2 = data2_147456[alu9];
    var val3 = data2_147456[(alu9+1536)];
    var val4 = data2_147456[(alu9+384)];
    var val5 = data2_147456[(alu9+768)];
    var val6 = data2_147456[(alu9+1920)];
    var val7 = data2_147456[(alu9+1152)];
    var val8 = data2_147456[(alu9+2688)];
    acc0[0] = (acc0[0]+(val0*val2));
    acc0[1] = (acc0[1]+(val0*val3));
    acc0[2] = (acc0[2]+(val0*val4));
    acc0[3] = (acc0[3]+(val0*val6));
    acc0[4] = (acc0[4]+(val0*val5));
    acc0[5] = (acc0[5]+(val0*val1));
    acc0[6] = (acc0[6]+(val0*val7));
    acc0[7] = (acc0[7]+(val0*val8));
  }
  var cast0 = bitcast<i32>((bitcast<u32>(gidx0)<<3u));
  var val9 = data3_384[cast0];
  var val10 = data3_384[(cast0+1)];
  var val11 = data3_384[(cast0+2)];
  var val12 = data3_384[(cast0+3)];
  var val13 = data3_384[(cast0+4)];
  var val14 = data3_384[(cast0+5)];
  var val15 = data3_384[(cast0+6)];
  var val16 = data3_384[(cast0+7)];
  var alu19 = (alu0+cast0);
  data0_222720[(alu19+1)] = (acc0[2]+val10);
  data0_222720[(alu19+2)] = (acc0[4]+val11);
  data0_222720[(alu19+3)] = (acc0[6]+val12);
  data0_222720[(alu19+4)] = (acc0[1]+val13);
  data0_222720[(alu19+5)] = (acc0[3]+val14);
  data0_222720[(alu19+6)] = (acc0[5]+val15);
  data0_222720[(alu19+7)] = (acc0[7]+val16);
  data0_222720[alu19] = (acc0[0]+val9);
}`;

const r_4_6_5_29_29_5_64 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_504600:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_222720:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_222720:array<f32>;
@compute @workgroup_size(29) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,5>;
  var gidx0 = i32(gindex.x); /* 145 */
  var gidx1 = i32(gindex.y); /* 6 */
  var gidx2 = i32(gindex.z); /* 4 */
  var lidx0 = i32(lindex.x); /* 29 */
  var alu0 = ((gidx0*71)>>11u);
  var alu1 = (gidx2*55680);
  var alu2 = (gidx0-(29*alu0));
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  acc0[4] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 64; Ridx0++) {
    var alu8 = (bitcast<i32>((bitcast<u32>(gidx1)<<6u))+Ridx0);
    var val0 = data1_222720[(alu8+(lidx0*384)+(alu0*11136)+alu1)];
    var alu9 = (alu8+(alu2*1920)+alu1);
    var val1 = data2_222720[alu9];
    var val2 = data2_222720[(alu9+384)];
    var val3 = data2_222720[(alu9+768)];
    var val4 = data2_222720[(alu9+1152)];
    var val5 = data2_222720[(alu9+1536)];
    acc0[0] = (acc0[0]+(val0*val1));
    acc0[1] = (acc0[1]+(val0*val2));
    acc0[2] = (acc0[2]+(val0*val3));
    acc0[3] = (acc0[3]+(val0*val4));
    acc0[4] = (acc0[4]+(val0*val5));
  }
  var alu16 = ((lidx0*145)+(alu0*4205)+(alu2*5)+(gidx1*21025)+(gidx2*126150));
  data0_504600[(alu16+1)] = (acc0[1]*0.125f);
  data0_504600[(alu16+2)] = (acc0[2]*0.125f);
  data0_504600[(alu16+3)] = (acc0[3]*0.125f);
  data0_504600[(alu16+4)] = (acc0[4]*0.125f);
  data0_504600[alu16] = (acc0[0]*0.125f);
}`;

const r_120_29_145 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_3480:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_504600:array<f32>;
@compute @workgroup_size(29) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 120 */
  var lidx0 = i32(lindex.x); /* 29 */
  acc0[0] = (f32(-INFINITY));
  for (var Ridx0 = 0; Ridx0 < 145; Ridx0++) {
    var val0 = data1_504600[((gidx0*4205)+(lidx0*145)+Ridx0)];
    var alu1 = select(acc0[0],val0,(acc0[0]<val0));
    acc0[0] = alu1;
  }
  data0_3480[(lidx0+(gidx0*29))] = acc0[0];
}`;

const r_145_8_3_145 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_3480:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_504600:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_3480:array<f32>;
@compute @workgroup_size(8,3) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 145 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 3 */
  var alu0 = (lidx0+(gidx0*24)+bitcast<i32>((bitcast<u32>(lidx1)<<3u)));
  var val0 = data2_3480[alu0];
  acc0[0] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 145; Ridx0++) {
    var val1 = data1_504600[((gidx0*3480)+(lidx1*1160)+(lidx0*145)+Ridx0)];
    acc0[0] = (acc0[0]+exp2(((val1-val0)*1.4426950408889634f)));
  }
  data0_3480[alu0] = acc0[0];
}`;

const r_4_145_16_4_6_145 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_222720:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_504600:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_3480:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_3480:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_222720:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,24>;
  var gidx0 = i32(gindex.x); /* 145 */
  var gidx1 = i32(gindex.y); /* 4 */
  var alu0 = (gidx0+(gidx1*870));
  var val0 = data2_3480[alu0];
  var alu1 = (alu0+145);
  var val1 = data2_3480[alu1];
  var alu2 = (alu0+290);
  var val2 = data2_3480[alu2];
  var alu3 = (alu0+435);
  var val3 = data2_3480[alu3];
  var alu4 = (alu0+580);
  var val4 = data2_3480[alu4];
  var alu5 = (alu0+725);
  var val5 = data2_3480[alu5];
  var lidx0 = i32(lindex.x); /* 16 */
  var alu6 = (gidx1*55680);
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  acc0[4] = 0.0f;
  acc0[5] = 0.0f;
  acc0[6] = 0.0f;
  acc0[7] = 0.0f;
  acc0[8] = 0.0f;
  acc0[9] = 0.0f;
  acc0[10] = 0.0f;
  acc0[11] = 0.0f;
  acc0[12] = 0.0f;
  acc0[13] = 0.0f;
  acc0[14] = 0.0f;
  acc0[15] = 0.0f;
  acc0[16] = 0.0f;
  acc0[17] = 0.0f;
  acc0[18] = 0.0f;
  acc0[19] = 0.0f;
  acc0[20] = 0.0f;
  acc0[21] = 0.0f;
  acc0[22] = 0.0f;
  acc0[23] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 145; Ridx0++) {
    var alu31 = ((gidx0*145)+Ridx0+(gidx1*126150));
    var val6 = data1_504600[alu31];
    var alu32 = (lidx0+(Ridx0*384)+alu6);
    var val7 = data4_222720[alu32];
    var val8 = data1_504600[(alu31+21025)];
    var val9 = data4_222720[(alu32+64)];
    var val10 = data1_504600[(alu31+42050)];
    var val11 = data4_222720[(alu32+32)];
    var val12 = data4_222720[(alu32+128)];
    var val13 = data1_504600[(alu31+63075)];
    var val14 = data4_222720[(alu32+192)];
    var val15 = data1_504600[(alu31+84100)];
    var val16 = data4_222720[(alu32+256)];
    var val17 = data1_504600[(alu31+105125)];
    var val18 = data4_222720[(alu32+320)];
    var val19 = data4_222720[(alu32+16)];
    var val20 = data4_222720[(alu32+80)];
    var val21 = data4_222720[(alu32+144)];
    var val22 = data4_222720[(alu32+208)];
    var val23 = data4_222720[(alu32+272)];
    var val24 = data4_222720[(alu32+336)];
    var val25 = data4_222720[(alu32+96)];
    var val26 = data4_222720[(alu32+160)];
    var val27 = data4_222720[(alu32+112)];
    var val28 = data4_222720[(alu32+288)];
    var val29 = data4_222720[(alu32+352)];
    var val30 = data4_222720[(alu32+48)];
    var val31 = data4_222720[(alu32+224)];
    var val32 = data4_222720[(alu32+176)];
    var val33 = data4_222720[(alu32+240)];
    var val34 = data4_222720[(alu32+304)];
    var val35 = data4_222720[(alu32+368)];
    var alu33 = exp2(((val8-val1)*1.4426950408889634f));
    var alu34 = exp2(((val10-val2)*1.4426950408889634f));
    var alu35 = exp2(((val13-val3)*1.4426950408889634f));
    var alu36 = exp2(((val15-val4)*1.4426950408889634f));
    var alu37 = exp2(((val17-val5)*1.4426950408889634f));
    var alu38 = exp2(((val6-val0)*1.4426950408889634f));
    acc0[0] = (acc0[0]+(alu38*val7));
    acc0[1] = (acc0[1]+(alu33*val9));
    acc0[2] = (acc0[2]+(alu34*val12));
    acc0[3] = (acc0[3]+(alu35*val14));
    acc0[4] = (acc0[4]+(alu36*val16));
    acc0[5] = (acc0[5]+(alu37*val18));
    acc0[6] = (acc0[6]+(alu38*val19));
    acc0[7] = (acc0[7]+(alu33*val20));
    acc0[8] = (acc0[8]+(alu34*val21));
    acc0[9] = (acc0[9]+(alu35*val22));
    acc0[10] = (acc0[10]+(alu36*val23));
    acc0[11] = (acc0[11]+(alu37*val24));
    acc0[12] = (acc0[12]+(alu38*val11));
    acc0[13] = (acc0[13]+(alu33*val25));
    acc0[14] = (acc0[14]+(alu34*val26));
    acc0[15] = (acc0[15]+(alu35*val31));
    acc0[16] = (acc0[16]+(alu36*val28));
    acc0[17] = (acc0[17]+(alu37*val29));
    acc0[18] = (acc0[18]+(alu38*val30));
    acc0[19] = (acc0[19]+(alu33*val27));
    acc0[20] = (acc0[20]+(alu34*val32));
    acc0[21] = (acc0[21]+(alu35*val33));
    acc0[22] = (acc0[22]+(alu36*val34));
    acc0[23] = (acc0[23]+(alu37*val35));
  }
  var val36 = data3_3480[alu0];
  var val37 = data3_3480[alu1];
  var val38 = data3_3480[alu2];
  var val39 = data3_3480[alu3];
  var val40 = data3_3480[alu4];
  var val41 = data3_3480[alu5];
  var alu64 = (lidx0+(gidx0*384)+alu6);
  var alu65 = (1/val36);
  var alu66 = (1/val37);
  var alu67 = (1/val38);
  var alu68 = (1/val39);
  var alu69 = (1/val40);
  var alu70 = (1/val41);
  data0_222720[alu64] = (acc0[0]*alu65);
  data0_222720[(alu64+16)] = (acc0[6]*alu65);
  data0_222720[(alu64+32)] = (acc0[12]*alu65);
  data0_222720[(alu64+48)] = (acc0[18]*alu65);
  data0_222720[(alu64+64)] = (acc0[1]*alu66);
  data0_222720[(alu64+80)] = (acc0[7]*alu66);
  data0_222720[(alu64+96)] = (acc0[13]*alu66);
  data0_222720[(alu64+112)] = (acc0[19]*alu66);
  data0_222720[(alu64+128)] = (acc0[2]*alu67);
  data0_222720[(alu64+144)] = (acc0[8]*alu67);
  data0_222720[(alu64+160)] = (acc0[14]*alu67);
  data0_222720[(alu64+176)] = (acc0[20]*alu67);
  data0_222720[(alu64+192)] = (acc0[3]*alu68);
  data0_222720[(alu64+208)] = (acc0[9]*alu68);
  data0_222720[(alu64+224)] = (acc0[15]*alu68);
  data0_222720[(alu64+240)] = (acc0[21]*alu68);
  data0_222720[(alu64+256)] = (acc0[4]*alu69);
  data0_222720[(alu64+272)] = (acc0[10]*alu69);
  data0_222720[(alu64+288)] = (acc0[16]*alu69);
  data0_222720[(alu64+304)] = (acc0[22]*alu69);
  data0_222720[(alu64+320)] = (acc0[5]*alu70);
  data0_222720[(alu64+336)] = (acc0[11]*alu70);
  data0_222720[(alu64+352)] = (acc0[17]*alu70);
  data0_222720[(alu64+368)] = (acc0[23]*alu70);
}`;

const r_2_2_5_96_29_4_384 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_222720:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_222720:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_147456:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_384:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_384:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_221568:array<f32>;
@compute @workgroup_size(29) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 480 */
  var gidx1 = i32(gindex.y); /* 2 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 29 */
  var alu0 = ((gidx1*55680)+(gidx2*111360));
  var alu1 = ((gidx0*171)>>14u);
  var alu2 = ((lidx0*384)+(alu1*11136));
  var alu3 = (gidx0-(96*alu1));
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 384; Ridx0++) {
    var val0 = data1_222720[(alu2+Ridx0+alu0)];
    var alu8 = ((alu3*1536)+Ridx0);
    var val1 = data2_147456[(alu8+768)];
    var val2 = data2_147456[alu8];
    var val3 = data2_147456[(alu8+384)];
    var val4 = data2_147456[(alu8+1152)];
    acc0[0] = (acc0[0]+(val0*val2));
    acc0[1] = (acc0[1]+(val0*val3));
    acc0[2] = (acc0[2]+(val0*val1));
    acc0[3] = (acc0[3]+(val0*val4));
  }
  var cast0 = bitcast<i32>((bitcast<u32>(alu3)<<2u));
  var val5 = data3_384[cast0];
  var val6 = data4_384[cast0];
  var val7 = data5_221568[cast0];
  var alu14 = (lidx0+(alu1*5)+11);
  var alu15 = ((alu14*43)>>9u);
  var alu16 = ((gidx1*4608)+((alu14-(12*alu15))*384)+(alu1*18432)+(alu15*9216)+(gidx2*110592)+cast0);
  var alu17 = (0<(lidx0+alu1));
  var val8 = select(0.0f, data5_221568[(alu16+-8832)], alu17);
  var alu18 = (cast0+1);
  var val9 = data3_384[alu18];
  var val10 = data4_384[alu18];
  var val11 = data5_221568[alu18];
  var val12 = select(0.0f, data5_221568[(alu16+-8831)], alu17);
  var alu19 = (cast0+2);
  var val13 = data3_384[alu19];
  var val14 = data4_384[alu19];
  var val15 = data5_221568[alu19];
  var val16 = select(0.0f, data5_221568[(alu16+-8830)], alu17);
  var alu20 = (cast0+3);
  var val17 = data3_384[alu20];
  var val18 = data4_384[alu20];
  var val19 = data5_221568[alu20];
  var val20 = select(0.0f, data5_221568[(alu16+-8829)], alu17);
  var alu21 = (alu2+cast0+alu0);
  var alu22 = ((lidx0+(alu1*29))<1);
  var alu23 = select(0.0f,val7,alu22);
  var alu24 = select(0.0f,val11,alu22);
  var alu25 = select(0.0f,val15,alu22);
  var alu26 = select(0.0f,val19,alu22);
  data0_222720[(alu21+1)] = (((acc0[1]+val9)*val10)+alu24+val12);
  data0_222720[(alu21+2)] = (((acc0[2]+val13)*val14)+alu25+val16);
  data0_222720[(alu21+3)] = (((acc0[3]+val17)*val18)+alu26+val20);
  data0_222720[alu21] = (((acc0[0]+val5)*val6)+alu23+val8);
}`;

const r_580_16_24 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32,16>;
@group(0) @binding(1)var<storage,read_write>data0_580:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_222720:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,1>;
  var acc1: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 580 */
  var lidx0 = i32(lindex.x); /* 16 */
  acc0[0] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 24; Ridx0++) {
    var val0 = data1_222720[((lidx0*24)+Ridx0+(gidx0*384))];
    acc0[0] = (acc0[0]+val0);
  }
  temp0[lidx0] = acc0[0];
  workgroupBarrier();
  acc1[0] = 0.0f;
  for (var Ridx102 = 0; Ridx102 < 16; Ridx102++) {
    var val1 = temp0[Ridx102];
    acc1[0] = (acc1[0]+val1);
  }
  var alu8 = (lidx0==0);
  if (alu8) {
    data0_580[gidx0] = (acc1[0]*0.0026041666666666665f);
  }
}`;

const r_580_16_24n1 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32,16>;
@group(0) @binding(1)var<storage,read_write>data0_580:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_222720:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_580:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,1>;
  var acc1: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 580 */
  var val0 = data2_580[gidx0];
  var lidx0 = i32(lindex.x); /* 16 */
  acc0[0] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 24; Ridx0++) {
    var val1 = data1_222720[((lidx0*24)+Ridx0+(gidx0*384))];
    var alu1 = (val1-val0);
    acc0[0] = (acc0[0]+(alu1*alu1));
  }
  temp0[lidx0] = acc0[0];
  workgroupBarrier();
  acc1[0] = 0.0f;
  for (var Ridx102 = 0; Ridx102 < 16; Ridx102++) {
    var val2 = temp0[Ridx102];
    acc1[0] = (acc1[0]+val2);
  }
  var alu9 = (lidx0==0);
  if (alu9) {
    data0_580[gidx0] = (1/sqrt(((acc1[0]*0.0026041666666666665f)+1e-05f)));
  }
}`;

const E_290_24_16_2 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_222720:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_222720:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_580:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_580:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_384:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_384:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 24 */
  var gidx1 = i32(gindex.y); /* 290 */
  var lidx0 = i32(lindex.x); /* 16 */
  var alu0 = (lidx0+bitcast<i32>((bitcast<u32>(gidx0)<<4u)));
  var alu1 = (alu0+(gidx1*768));
  var val0 = data1_222720[alu1];
  var cast0 = bitcast<i32>((bitcast<u32>(gidx1)<<1u));
  var val1 = data2_580[cast0];
  var val2 = data3_580[cast0];
  var val3 = data4_384[alu0];
  var val4 = data5_384[alu0];
  var alu2 = (alu1+384);
  var val5 = data1_222720[alu2];
  var alu3 = (cast0+1);
  var val6 = data2_580[alu3];
  var val7 = data3_580[alu3];
  data0_222720[alu1] = (((val0-val1)*val2*val3)+val4);
  data0_222720[alu2] = (((val5-val6)*val7*val3)+val4);
}`;

const r_20_128_29_4_3_384 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_890880:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_222720:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_589824:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_1536:array<f32>;
@compute @workgroup_size(29) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,12>;
  var gidx0 = i32(gindex.x); /* 128 */
  var gidx1 = i32(gindex.y); /* 20 */
  var lidx0 = i32(lindex.x); /* 29 */
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  acc0[4] = 0.0f;
  acc0[5] = 0.0f;
  acc0[6] = 0.0f;
  acc0[7] = 0.0f;
  acc0[8] = 0.0f;
  acc0[9] = 0.0f;
  acc0[10] = 0.0f;
  acc0[11] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 384; Ridx0++) {
    var val0 = data1_222720[((gidx1*11136)+(lidx0*384)+Ridx0)];
    var alu12 = ((gidx0*4608)+Ridx0);
    var val1 = data2_589824[alu12];
    var val2 = data2_589824[(alu12+1536)];
    var val3 = data2_589824[(alu12+3072)];
    var val4 = data2_589824[(alu12+384)];
    var val5 = data2_589824[(alu12+1920)];
    var val6 = data2_589824[(alu12+768)];
    var val7 = data2_589824[(alu12+2304)];
    var val8 = data2_589824[(alu12+3840)];
    var val9 = data2_589824[(alu12+1152)];
    var val10 = data2_589824[(alu12+3456)];
    var val11 = data2_589824[(alu12+2688)];
    var val12 = data2_589824[(alu12+4224)];
    acc0[0] = (acc0[0]+(val0*val1));
    acc0[1] = (acc0[1]+(val0*val2));
    acc0[2] = (acc0[2]+(val0*val3));
    acc0[3] = (acc0[3]+(val0*val4));
    acc0[4] = (acc0[4]+(val0*val5));
    acc0[5] = (acc0[5]+(val0*val10));
    acc0[6] = (acc0[6]+(val0*val6));
    acc0[7] = (acc0[7]+(val0*val7));
    acc0[8] = (acc0[8]+(val0*val8));
    acc0[9] = (acc0[9]+(val0*val9));
    acc0[10] = (acc0[10]+(val0*val11));
    acc0[11] = (acc0[11]+(val0*val12));
  }
  var alu26 = (gidx0*12);
  var val13 = data3_1536[alu26];
  var val14 = data3_1536[(alu26+1)];
  var val15 = data3_1536[(alu26+2)];
  var val16 = data3_1536[(alu26+3)];
  var val17 = data3_1536[(alu26+4)];
  var val18 = data3_1536[(alu26+5)];
  var val19 = data3_1536[(alu26+6)];
  var val20 = data3_1536[(alu26+7)];
  var val21 = data3_1536[(alu26+8)];
  var val22 = data3_1536[(alu26+9)];
  var val23 = data3_1536[(alu26+10)];
  var val24 = data3_1536[(alu26+11)];
  var alu27 = ((gidx1*44544)+(lidx0*1536)+alu26);
  var alu28 = (acc0[0]+val13);
  var alu29 = (acc0[1]+val17);
  var alu30 = (acc0[2]+val21);
  var alu31 = (acc0[3]+val14);
  var alu32 = (acc0[4]+val18);
  var alu33 = (acc0[5]+val22);
  var alu34 = (acc0[6]+val15);
  var alu35 = (acc0[7]+val19);
  var alu36 = (acc0[8]+val23);
  var alu37 = (acc0[9]+val16);
  var alu38 = (acc0[10]+val20);
  var alu39 = (acc0[11]+val24);
  var alu40 = (alu28*0.7071067811865475f);
  var alu41 = select(1.0f,-1.0f,(alu40<0.0f));
  var alu42 = select(0.0f,alu41,(alu40!=0.0f));
  var alu43 = (1/(1.0f+(alu28*alu42*0.2316418882663604f)));
  var alu44 = (alu29*0.7071067811865475f);
  var alu45 = select(1.0f,-1.0f,(alu44<0.0f));
  var alu46 = select(0.0f,alu45,(alu44!=0.0f));
  var alu47 = (1/(1.0f+(alu29*alu46*0.2316418882663604f)));
  var alu48 = (alu30*0.7071067811865475f);
  var alu49 = select(1.0f,-1.0f,(alu48<0.0f));
  var alu50 = select(0.0f,alu49,(alu48!=0.0f));
  var alu51 = (1/(1.0f+(alu30*alu50*0.2316418882663604f)));
  var alu52 = (alu31*0.7071067811865475f);
  var alu53 = select(1.0f,-1.0f,(alu52<0.0f));
  var alu54 = select(0.0f,alu53,(alu52!=0.0f));
  var alu55 = (1/(1.0f+(alu31*alu54*0.2316418882663604f)));
  var alu56 = (alu32*0.7071067811865475f);
  var alu57 = select(1.0f,-1.0f,(alu56<0.0f));
  var alu58 = select(0.0f,alu57,(alu56!=0.0f));
  var alu59 = (1/(1.0f+(alu32*alu58*0.2316418882663604f)));
  var alu60 = (alu33*0.7071067811865475f);
  var alu61 = select(1.0f,-1.0f,(alu60<0.0f));
  var alu62 = select(0.0f,alu61,(alu60!=0.0f));
  var alu63 = (1/(1.0f+(alu33*alu62*0.2316418882663604f)));
  var alu64 = (alu34*0.7071067811865475f);
  var alu65 = select(1.0f,-1.0f,(alu64<0.0f));
  var alu66 = select(0.0f,alu65,(alu64!=0.0f));
  var alu67 = (1/(1.0f+(alu34*alu66*0.2316418882663604f)));
  var alu68 = (alu35*0.7071067811865475f);
  var alu69 = select(1.0f,-1.0f,(alu68<0.0f));
  var alu70 = select(0.0f,alu69,(alu68!=0.0f));
  var alu71 = (1/(1.0f+(alu35*alu70*0.2316418882663604f)));
  var alu72 = (alu36*0.7071067811865475f);
  var alu73 = select(1.0f,-1.0f,(alu72<0.0f));
  var alu74 = select(0.0f,alu73,(alu72!=0.0f));
  var alu75 = (1/(1.0f+(alu36*alu74*0.2316418882663604f)));
  var alu76 = (alu37*0.7071067811865475f);
  var alu77 = select(1.0f,-1.0f,(alu76<0.0f));
  var alu78 = select(0.0f,alu77,(alu76!=0.0f));
  var alu79 = (1/(1.0f+(alu37*alu78*0.2316418882663604f)));
  var alu80 = (alu38*0.7071067811865475f);
  var alu81 = select(1.0f,-1.0f,(alu80<0.0f));
  var alu82 = select(0.0f,alu81,(alu80!=0.0f));
  var alu83 = (1/(1.0f+(alu38*alu82*0.2316418882663604f)));
  var alu84 = (alu39*0.7071067811865475f);
  var alu85 = select(1.0f,-1.0f,(alu84<0.0f));
  var alu86 = select(0.0f,alu85,(alu84!=0.0f));
  var alu87 = (1/(1.0f+(alu39*alu86*0.2316418882663604f)));
  data0_890880[(alu27+1)] = (alu31*(1.0f+(alu54*(1.0f-(alu55*((((((((1.061405429f*alu55)+-1.453152027f)*alu55)+1.421413741f)*alu55)+-0.284496736f)*alu55)+0.254829592f)*exp2((alu31*alu31*-0.7213475204444816f))))))*0.5f);
  data0_890880[(alu27+2)] = (alu34*(1.0f+(alu66*(1.0f-(alu67*((((((((1.061405429f*alu67)+-1.453152027f)*alu67)+1.421413741f)*alu67)+-0.284496736f)*alu67)+0.254829592f)*exp2((alu34*alu34*-0.7213475204444816f))))))*0.5f);
  data0_890880[(alu27+3)] = (alu37*(1.0f+(alu78*(1.0f-(alu79*((((((((1.061405429f*alu79)+-1.453152027f)*alu79)+1.421413741f)*alu79)+-0.284496736f)*alu79)+0.254829592f)*exp2((alu37*alu37*-0.7213475204444816f))))))*0.5f);
  data0_890880[(alu27+4)] = (alu29*(1.0f+(alu46*(1.0f-(alu47*((((((((1.061405429f*alu47)+-1.453152027f)*alu47)+1.421413741f)*alu47)+-0.284496736f)*alu47)+0.254829592f)*exp2((alu29*alu29*-0.7213475204444816f))))))*0.5f);
  data0_890880[(alu27+5)] = (alu32*(1.0f+(alu58*(1.0f-(alu59*((((((((1.061405429f*alu59)+-1.453152027f)*alu59)+1.421413741f)*alu59)+-0.284496736f)*alu59)+0.254829592f)*exp2((alu32*alu32*-0.7213475204444816f))))))*0.5f);
  data0_890880[(alu27+6)] = (alu35*(1.0f+(alu70*(1.0f-(alu71*((((((((1.061405429f*alu71)+-1.453152027f)*alu71)+1.421413741f)*alu71)+-0.284496736f)*alu71)+0.254829592f)*exp2((alu35*alu35*-0.7213475204444816f))))))*0.5f);
  data0_890880[(alu27+7)] = (alu38*(1.0f+(alu82*(1.0f-(alu83*((((((((1.061405429f*alu83)+-1.453152027f)*alu83)+1.421413741f)*alu83)+-0.284496736f)*alu83)+0.254829592f)*exp2((alu38*alu38*-0.7213475204444816f))))))*0.5f);
  data0_890880[(alu27+8)] = (alu30*(1.0f+(alu50*(1.0f-(alu51*((((((((1.061405429f*alu51)+-1.453152027f)*alu51)+1.421413741f)*alu51)+-0.284496736f)*alu51)+0.254829592f)*exp2((alu30*alu30*-0.7213475204444816f))))))*0.5f);
  data0_890880[(alu27+9)] = (alu33*(1.0f+(alu62*(1.0f-(alu63*((((((((1.061405429f*alu63)+-1.453152027f)*alu63)+1.421413741f)*alu63)+-0.284496736f)*alu63)+0.254829592f)*exp2((alu33*alu33*-0.7213475204444816f))))))*0.5f);
  data0_890880[(alu27+10)] = (alu36*(1.0f+(alu74*(1.0f-(alu75*((((((((1.061405429f*alu75)+-1.453152027f)*alu75)+1.421413741f)*alu75)+-0.284496736f)*alu75)+0.254829592f)*exp2((alu36*alu36*-0.7213475204444816f))))))*0.5f);
  data0_890880[(alu27+11)] = (alu39*(1.0f+(alu86*(1.0f-(alu87*((((((((1.061405429f*alu87)+-1.453152027f)*alu87)+1.421413741f)*alu87)+-0.284496736f)*alu87)+0.254829592f)*exp2((alu39*alu39*-0.7213475204444816f))))))*0.5f);
  data0_890880[alu27] = (alu28*(1.0f+(alu42*(1.0f-(alu43*((((((((1.061405429f*alu43)+-1.453152027f)*alu43)+1.421413741f)*alu43)+-0.284496736f)*alu43)+0.254829592f)*exp2((alu28*alu28*-0.7213475204444816f))))))*0.5f);
}`;

const r_5_32_29_4_3_4_1536 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_222720:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_890880:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_589824:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_384:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_384:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_222720:array<f32>;
@compute @workgroup_size(29) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,48>;
  var gidx0 = i32(gindex.x); /* 32 */
  var gidx1 = i32(gindex.y); /* 5 */
  var lidx0 = i32(lindex.x); /* 29 */
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  acc0[4] = 0.0f;
  acc0[5] = 0.0f;
  acc0[6] = 0.0f;
  acc0[7] = 0.0f;
  acc0[8] = 0.0f;
  acc0[9] = 0.0f;
  acc0[10] = 0.0f;
  acc0[11] = 0.0f;
  acc0[12] = 0.0f;
  acc0[13] = 0.0f;
  acc0[14] = 0.0f;
  acc0[15] = 0.0f;
  acc0[16] = 0.0f;
  acc0[17] = 0.0f;
  acc0[18] = 0.0f;
  acc0[19] = 0.0f;
  acc0[20] = 0.0f;
  acc0[21] = 0.0f;
  acc0[22] = 0.0f;
  acc0[23] = 0.0f;
  acc0[24] = 0.0f;
  acc0[25] = 0.0f;
  acc0[26] = 0.0f;
  acc0[27] = 0.0f;
  acc0[28] = 0.0f;
  acc0[29] = 0.0f;
  acc0[30] = 0.0f;
  acc0[31] = 0.0f;
  acc0[32] = 0.0f;
  acc0[33] = 0.0f;
  acc0[34] = 0.0f;
  acc0[35] = 0.0f;
  acc0[36] = 0.0f;
  acc0[37] = 0.0f;
  acc0[38] = 0.0f;
  acc0[39] = 0.0f;
  acc0[40] = 0.0f;
  acc0[41] = 0.0f;
  acc0[42] = 0.0f;
  acc0[43] = 0.0f;
  acc0[44] = 0.0f;
  acc0[45] = 0.0f;
  acc0[46] = 0.0f;
  acc0[47] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 1536; Ridx0++) {
    var alu48 = ((gidx1*178176)+(lidx0*1536)+Ridx0);
    var val0 = data1_890880[(alu48+89088)];
    var val1 = data1_890880[alu48];
    var alu49 = ((gidx0*18432)+Ridx0);
    var val2 = data2_589824[alu49];
    var val3 = data1_890880[(alu48+44544)];
    var val4 = data1_890880[(alu48+133632)];
    var val5 = data2_589824[(alu49+6144)];
    var val6 = data2_589824[(alu49+12288)];
    var val7 = data2_589824[(alu49+1536)];
    var val8 = data2_589824[(alu49+7680)];
    var val9 = data2_589824[(alu49+13824)];
    var val10 = data2_589824[(alu49+3072)];
    var val11 = data2_589824[(alu49+9216)];
    var val12 = data2_589824[(alu49+15360)];
    var val13 = data2_589824[(alu49+4608)];
    var val14 = data2_589824[(alu49+10752)];
    var val15 = data2_589824[(alu49+16896)];
    acc0[0] = (acc0[0]+(val1*val2));
    acc0[1] = (acc0[1]+(val3*val2));
    acc0[2] = (acc0[2]+(val0*val2));
    acc0[3] = (acc0[3]+(val4*val2));
    acc0[4] = (acc0[4]+(val1*val5));
    acc0[5] = (acc0[5]+(val3*val5));
    acc0[6] = (acc0[6]+(val0*val5));
    acc0[7] = (acc0[7]+(val4*val5));
    acc0[8] = (acc0[8]+(val1*val6));
    acc0[9] = (acc0[9]+(val3*val6));
    acc0[10] = (acc0[10]+(val0*val6));
    acc0[11] = (acc0[11]+(val4*val6));
    acc0[12] = (acc0[12]+(val1*val7));
    acc0[13] = (acc0[13]+(val3*val7));
    acc0[14] = (acc0[14]+(val0*val7));
    acc0[15] = (acc0[15]+(val4*val7));
    acc0[16] = (acc0[16]+(val1*val8));
    acc0[17] = (acc0[17]+(val3*val8));
    acc0[18] = (acc0[18]+(val0*val8));
    acc0[19] = (acc0[19]+(val4*val8));
    acc0[20] = (acc0[20]+(val1*val9));
    acc0[21] = (acc0[21]+(val3*val9));
    acc0[22] = (acc0[22]+(val0*val9));
    acc0[23] = (acc0[23]+(val4*val9));
    acc0[24] = (acc0[24]+(val1*val10));
    acc0[25] = (acc0[25]+(val3*val10));
    acc0[26] = (acc0[26]+(val0*val10));
    acc0[27] = (acc0[27]+(val4*val10));
    acc0[28] = (acc0[28]+(val1*val11));
    acc0[29] = (acc0[29]+(val3*val11));
    acc0[30] = (acc0[30]+(val0*val11));
    acc0[31] = (acc0[31]+(val4*val11));
    acc0[32] = (acc0[32]+(val1*val12));
    acc0[33] = (acc0[33]+(val3*val12));
    acc0[34] = (acc0[34]+(val0*val12));
    acc0[35] = (acc0[35]+(val4*val12));
    acc0[36] = (acc0[36]+(val1*val13));
    acc0[37] = (acc0[37]+(val3*val13));
    acc0[38] = (acc0[38]+(val0*val13));
    acc0[39] = (acc0[39]+(val4*val13));
    acc0[40] = (acc0[40]+(val1*val14));
    acc0[41] = (acc0[41]+(val3*val14));
    acc0[42] = (acc0[42]+(val0*val14));
    acc0[43] = (acc0[43]+(val4*val14));
    acc0[44] = (acc0[44]+(val1*val15));
    acc0[45] = (acc0[45]+(val3*val15));
    acc0[46] = (acc0[46]+(val0*val15));
    acc0[47] = (acc0[47]+(val4*val15));
  }
  var alu99 = (gidx0*12);
  var val16 = data3_384[alu99];
  var val17 = data4_384[alu99];
  var alu100 = ((gidx1*44544)+(lidx0*384)+alu99);
  var val18 = data5_222720[alu100];
  var alu101 = (alu99+1);
  var val19 = data3_384[alu101];
  var val20 = data4_384[alu101];
  var alu102 = (alu100+1);
  var val21 = data5_222720[alu102];
  var alu103 = (alu99+2);
  var val22 = data3_384[alu103];
  var val23 = data4_384[alu103];
  var alu104 = (alu100+2);
  var val24 = data5_222720[alu104];
  var alu105 = (alu99+3);
  var val25 = data3_384[alu105];
  var val26 = data4_384[alu105];
  var alu106 = (alu100+3);
  var val27 = data5_222720[alu106];
  var alu107 = (alu99+4);
  var val28 = data3_384[alu107];
  var alu108 = (alu99+5);
  var val29 = data3_384[alu108];
  var val30 = data4_384[alu107];
  var alu109 = (alu100+4);
  var val31 = data5_222720[alu109];
  var val32 = data4_384[alu108];
  var alu110 = (alu100+5);
  var val33 = data5_222720[alu110];
  var alu111 = (alu99+6);
  var val34 = data3_384[alu111];
  var alu112 = (alu99+7);
  var val35 = data3_384[alu112];
  var val36 = data4_384[alu111];
  var alu113 = (alu100+6);
  var val37 = data5_222720[alu113];
  var val38 = data4_384[alu112];
  var alu114 = (alu100+7);
  var val39 = data5_222720[alu114];
  var alu115 = (alu99+8);
  var val40 = data3_384[alu115];
  var val41 = data4_384[alu115];
  var alu116 = (alu100+8);
  var val42 = data5_222720[alu116];
  var alu117 = (alu99+9);
  var val43 = data3_384[alu117];
  var val44 = data4_384[alu117];
  var alu118 = (alu100+9);
  var val45 = data5_222720[alu118];
  var alu119 = (alu99+10);
  var val46 = data3_384[alu119];
  var val47 = data4_384[alu119];
  var alu120 = (alu100+10);
  var val48 = data5_222720[alu120];
  var alu121 = (alu99+11);
  var val49 = data3_384[alu121];
  var val50 = data4_384[alu121];
  var alu122 = (alu100+11);
  var val51 = data5_222720[alu122];
  var alu123 = (alu100+11136);
  var val52 = data5_222720[alu123];
  var alu124 = (alu100+11137);
  var val53 = data5_222720[alu124];
  var alu125 = (alu100+11138);
  var val54 = data5_222720[alu125];
  var alu126 = (alu100+11139);
  var val55 = data5_222720[alu126];
  var alu127 = (alu100+11140);
  var val56 = data5_222720[alu127];
  var alu128 = (alu100+11141);
  var val57 = data5_222720[alu128];
  var alu129 = (alu100+11142);
  var val58 = data5_222720[alu129];
  var alu130 = (alu100+11143);
  var val59 = data5_222720[alu130];
  var alu131 = (alu100+11144);
  var val60 = data5_222720[alu131];
  var alu132 = (alu100+11145);
  var val61 = data5_222720[alu132];
  var alu133 = (alu100+11146);
  var val62 = data5_222720[alu133];
  var alu134 = (alu100+11147);
  var val63 = data5_222720[alu134];
  var alu135 = (alu100+22272);
  var val64 = data5_222720[alu135];
  var alu136 = (alu100+22273);
  var val65 = data5_222720[alu136];
  var alu137 = (alu100+22274);
  var val66 = data5_222720[alu137];
  var alu138 = (alu100+22275);
  var val67 = data5_222720[alu138];
  var alu139 = (alu100+22276);
  var val68 = data5_222720[alu139];
  var alu140 = (alu100+22277);
  var val69 = data5_222720[alu140];
  var alu141 = (alu100+22278);
  var val70 = data5_222720[alu141];
  var alu142 = (alu100+22279);
  var val71 = data5_222720[alu142];
  var alu143 = (alu100+22280);
  var val72 = data5_222720[alu143];
  var alu144 = (alu100+22281);
  var val73 = data5_222720[alu144];
  var alu145 = (alu100+22282);
  var val74 = data5_222720[alu145];
  var alu146 = (alu100+22283);
  var val75 = data5_222720[alu146];
  var alu147 = (alu100+33408);
  var val76 = data5_222720[alu147];
  var alu148 = (alu100+33409);
  var val77 = data5_222720[alu148];
  var alu149 = (alu100+33410);
  var val78 = data5_222720[alu149];
  var alu150 = (alu100+33411);
  var val79 = data5_222720[alu150];
  var alu151 = (alu100+33412);
  var val80 = data5_222720[alu151];
  var alu152 = (alu100+33413);
  var val81 = data5_222720[alu152];
  var alu153 = (alu100+33414);
  var val82 = data5_222720[alu153];
  var alu154 = (alu100+33415);
  var val83 = data5_222720[alu154];
  var alu155 = (alu100+33416);
  var val84 = data5_222720[alu155];
  var alu156 = (alu100+33417);
  var val85 = data5_222720[alu156];
  var alu157 = (alu100+33418);
  var val86 = data5_222720[alu157];
  var alu158 = (alu100+33419);
  var val87 = data5_222720[alu158];
  data0_222720[alu123] = (((acc0[1]+val16)*val17)+val52);
  data0_222720[alu124] = (((acc0[13]+val19)*val20)+val53);
  data0_222720[alu125] = (((acc0[25]+val22)*val23)+val54);
  data0_222720[alu126] = (((acc0[37]+val25)*val26)+val55);
  data0_222720[alu127] = (((acc0[5]+val28)*val30)+val56);
  data0_222720[alu128] = (((acc0[17]+val29)*val32)+val57);
  data0_222720[alu129] = (((acc0[29]+val34)*val36)+val58);
  data0_222720[alu130] = (((acc0[41]+val35)*val38)+val59);
  data0_222720[alu131] = (((acc0[9]+val40)*val41)+val60);
  data0_222720[alu132] = (((acc0[21]+val43)*val44)+val61);
  data0_222720[alu133] = (((acc0[33]+val46)*val47)+val62);
  data0_222720[alu134] = (((acc0[45]+val49)*val50)+val63);
  data0_222720[alu135] = (((acc0[2]+val16)*val17)+val64);
  data0_222720[alu136] = (((acc0[14]+val19)*val20)+val65);
  data0_222720[alu137] = (((acc0[26]+val22)*val23)+val66);
  data0_222720[alu138] = (((acc0[38]+val25)*val26)+val67);
  data0_222720[alu139] = (((acc0[6]+val28)*val30)+val68);
  data0_222720[alu140] = (((acc0[18]+val29)*val32)+val69);
  data0_222720[alu141] = (((acc0[30]+val34)*val36)+val70);
  data0_222720[alu142] = (((acc0[42]+val35)*val38)+val71);
  data0_222720[alu143] = (((acc0[10]+val40)*val41)+val72);
  data0_222720[alu144] = (((acc0[22]+val43)*val44)+val73);
  data0_222720[alu145] = (((acc0[34]+val46)*val47)+val74);
  data0_222720[alu146] = (((acc0[46]+val49)*val50)+val75);
  data0_222720[alu147] = (((acc0[3]+val16)*val17)+val76);
  data0_222720[alu148] = (((acc0[15]+val19)*val20)+val77);
  data0_222720[alu149] = (((acc0[27]+val22)*val23)+val78);
  data0_222720[alu150] = (((acc0[39]+val25)*val26)+val79);
  data0_222720[alu151] = (((acc0[7]+val28)*val30)+val80);
  data0_222720[alu152] = (((acc0[19]+val29)*val32)+val81);
  data0_222720[alu153] = (((acc0[31]+val34)*val36)+val82);
  data0_222720[alu154] = (((acc0[43]+val35)*val38)+val83);
  data0_222720[alu155] = (((acc0[11]+val40)*val41)+val84);
  data0_222720[alu156] = (((acc0[23]+val43)*val44)+val85);
  data0_222720[alu157] = (((acc0[35]+val46)*val47)+val86);
  data0_222720[alu158] = (((acc0[47]+val49)*val50)+val87);
  data0_222720[alu102] = (((acc0[12]+val19)*val20)+val21);
  data0_222720[alu104] = (((acc0[24]+val22)*val23)+val24);
  data0_222720[alu106] = (((acc0[36]+val25)*val26)+val27);
  data0_222720[alu109] = (((acc0[4]+val28)*val30)+val31);
  data0_222720[alu110] = (((acc0[16]+val29)*val32)+val33);
  data0_222720[alu113] = (((acc0[28]+val34)*val36)+val37);
  data0_222720[alu114] = (((acc0[40]+val35)*val38)+val39);
  data0_222720[alu116] = (((acc0[8]+val40)*val41)+val42);
  data0_222720[alu118] = (((acc0[20]+val43)*val44)+val45);
  data0_222720[alu120] = (((acc0[32]+val46)*val47)+val48);
  data0_222720[alu122] = (((acc0[44]+val49)*val50)+val51);
  data0_222720[alu100] = (((acc0[0]+val16)*val17)+val18);
}`;

const r_5_32_29_4_3_4_384 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_222720:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_222720:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_147456:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_384:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_384:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_222720:array<f32>;
@compute @workgroup_size(29) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,48>;
  var gidx0 = i32(gindex.x); /* 32 */
  var gidx1 = i32(gindex.y); /* 5 */
  var lidx0 = i32(lindex.x); /* 29 */
  var alu0 = ((gidx1*44544)+(lidx0*384));
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  acc0[4] = 0.0f;
  acc0[5] = 0.0f;
  acc0[6] = 0.0f;
  acc0[7] = 0.0f;
  acc0[8] = 0.0f;
  acc0[9] = 0.0f;
  acc0[10] = 0.0f;
  acc0[11] = 0.0f;
  acc0[12] = 0.0f;
  acc0[13] = 0.0f;
  acc0[14] = 0.0f;
  acc0[15] = 0.0f;
  acc0[16] = 0.0f;
  acc0[17] = 0.0f;
  acc0[18] = 0.0f;
  acc0[19] = 0.0f;
  acc0[20] = 0.0f;
  acc0[21] = 0.0f;
  acc0[22] = 0.0f;
  acc0[23] = 0.0f;
  acc0[24] = 0.0f;
  acc0[25] = 0.0f;
  acc0[26] = 0.0f;
  acc0[27] = 0.0f;
  acc0[28] = 0.0f;
  acc0[29] = 0.0f;
  acc0[30] = 0.0f;
  acc0[31] = 0.0f;
  acc0[32] = 0.0f;
  acc0[33] = 0.0f;
  acc0[34] = 0.0f;
  acc0[35] = 0.0f;
  acc0[36] = 0.0f;
  acc0[37] = 0.0f;
  acc0[38] = 0.0f;
  acc0[39] = 0.0f;
  acc0[40] = 0.0f;
  acc0[41] = 0.0f;
  acc0[42] = 0.0f;
  acc0[43] = 0.0f;
  acc0[44] = 0.0f;
  acc0[45] = 0.0f;
  acc0[46] = 0.0f;
  acc0[47] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 384; Ridx0++) {
    var alu49 = (alu0+Ridx0);
    var val0 = data1_222720[alu49];
    var alu50 = ((gidx0*4608)+Ridx0);
    var val1 = data2_147456[alu50];
    var val2 = data1_222720[(alu49+11136)];
    var val3 = data1_222720[(alu49+22272)];
    var val4 = data1_222720[(alu49+33408)];
    var val5 = data2_147456[(alu50+1536)];
    var val6 = data2_147456[(alu50+3072)];
    var val7 = data2_147456[(alu50+384)];
    var val8 = data2_147456[(alu50+1920)];
    var val9 = data2_147456[(alu50+3456)];
    var val10 = data2_147456[(alu50+768)];
    var val11 = data2_147456[(alu50+2304)];
    var val12 = data2_147456[(alu50+3840)];
    var val13 = data2_147456[(alu50+1152)];
    var val14 = data2_147456[(alu50+2688)];
    var val15 = data2_147456[(alu50+4224)];
    acc0[0] = (acc0[0]+(val0*val1));
    acc0[1] = (acc0[1]+(val2*val1));
    acc0[2] = (acc0[2]+(val3*val1));
    acc0[3] = (acc0[3]+(val4*val1));
    acc0[4] = (acc0[4]+(val0*val5));
    acc0[5] = (acc0[5]+(val2*val5));
    acc0[6] = (acc0[6]+(val3*val5));
    acc0[7] = (acc0[7]+(val4*val5));
    acc0[8] = (acc0[8]+(val0*val6));
    acc0[9] = (acc0[9]+(val2*val6));
    acc0[10] = (acc0[10]+(val3*val6));
    acc0[11] = (acc0[11]+(val4*val6));
    acc0[12] = (acc0[12]+(val0*val7));
    acc0[13] = (acc0[13]+(val2*val7));
    acc0[14] = (acc0[14]+(val3*val7));
    acc0[15] = (acc0[15]+(val4*val7));
    acc0[16] = (acc0[16]+(val0*val8));
    acc0[17] = (acc0[17]+(val2*val8));
    acc0[18] = (acc0[18]+(val3*val8));
    acc0[19] = (acc0[19]+(val4*val8));
    acc0[20] = (acc0[20]+(val0*val9));
    acc0[21] = (acc0[21]+(val2*val9));
    acc0[22] = (acc0[22]+(val3*val9));
    acc0[23] = (acc0[23]+(val4*val9));
    acc0[24] = (acc0[24]+(val0*val10));
    acc0[25] = (acc0[25]+(val2*val10));
    acc0[26] = (acc0[26]+(val3*val10));
    acc0[27] = (acc0[27]+(val4*val10));
    acc0[28] = (acc0[28]+(val0*val11));
    acc0[29] = (acc0[29]+(val2*val11));
    acc0[30] = (acc0[30]+(val3*val11));
    acc0[31] = (acc0[31]+(val4*val11));
    acc0[32] = (acc0[32]+(val0*val12));
    acc0[33] = (acc0[33]+(val2*val12));
    acc0[34] = (acc0[34]+(val3*val12));
    acc0[35] = (acc0[35]+(val4*val12));
    acc0[36] = (acc0[36]+(val0*val13));
    acc0[37] = (acc0[37]+(val2*val13));
    acc0[38] = (acc0[38]+(val3*val13));
    acc0[39] = (acc0[39]+(val4*val13));
    acc0[40] = (acc0[40]+(val0*val14));
    acc0[41] = (acc0[41]+(val2*val14));
    acc0[42] = (acc0[42]+(val3*val14));
    acc0[43] = (acc0[43]+(val4*val14));
    acc0[44] = (acc0[44]+(val0*val15));
    acc0[45] = (acc0[45]+(val2*val15));
    acc0[46] = (acc0[46]+(val3*val15));
    acc0[47] = (acc0[47]+(val4*val15));
  }
  var alu100 = (gidx0*12);
  var alu101 = (alu100+1);
  var val16 = data3_384[alu101];
  var val17 = data3_384[alu100];
  var val18 = data4_384[alu101];
  var val19 = data4_384[alu100];
  var alu102 = (alu0+alu100);
  var alu103 = (alu102+1);
  var val20 = data5_222720[alu103];
  var val21 = data5_222720[alu102];
  var alu104 = (alu100+2);
  var val22 = data3_384[alu104];
  var alu105 = (alu100+3);
  var val23 = data3_384[alu105];
  var alu106 = (alu100+4);
  var val24 = data3_384[alu106];
  var alu107 = (alu100+5);
  var val25 = data3_384[alu107];
  var val26 = data4_384[alu104];
  var alu108 = (alu102+2);
  var val27 = data5_222720[alu108];
  var val28 = data4_384[alu105];
  var alu109 = (alu102+3);
  var val29 = data5_222720[alu109];
  var val30 = data4_384[alu106];
  var alu110 = (alu102+4);
  var val31 = data5_222720[alu110];
  var val32 = data4_384[alu107];
  var alu111 = (alu102+5);
  var val33 = data5_222720[alu111];
  var alu112 = (alu100+6);
  var val34 = data3_384[alu112];
  var val35 = data4_384[alu112];
  var alu113 = (alu102+6);
  var val36 = data5_222720[alu113];
  var alu114 = (alu100+7);
  var val37 = data3_384[alu114];
  var val38 = data4_384[alu114];
  var alu115 = (alu102+7);
  var val39 = data5_222720[alu115];
  var alu116 = (alu100+8);
  var val40 = data3_384[alu116];
  var val41 = data4_384[alu116];
  var alu117 = (alu102+8);
  var val42 = data5_222720[alu117];
  var alu118 = (alu100+9);
  var val43 = data3_384[alu118];
  var val44 = data4_384[alu118];
  var alu119 = (alu102+9);
  var val45 = data5_222720[alu119];
  var alu120 = (alu100+10);
  var val46 = data3_384[alu120];
  var val47 = data4_384[alu120];
  var alu121 = (alu102+10);
  var val48 = data5_222720[alu121];
  var alu122 = (alu100+11);
  var val49 = data3_384[alu122];
  var val50 = data4_384[alu122];
  var alu123 = (alu102+11);
  var val51 = data5_222720[alu123];
  var alu124 = (alu102+11136);
  var val52 = data5_222720[alu124];
  var alu125 = (alu102+11137);
  var val53 = data5_222720[alu125];
  var alu126 = (alu102+11138);
  var val54 = data5_222720[alu126];
  var alu127 = (alu102+11139);
  var val55 = data5_222720[alu127];
  var alu128 = (alu102+11140);
  var val56 = data5_222720[alu128];
  var alu129 = (alu102+11141);
  var val57 = data5_222720[alu129];
  var alu130 = (alu102+11142);
  var val58 = data5_222720[alu130];
  var alu131 = (alu102+11143);
  var val59 = data5_222720[alu131];
  var alu132 = (alu102+11144);
  var val60 = data5_222720[alu132];
  var alu133 = (alu102+11145);
  var val61 = data5_222720[alu133];
  var alu134 = (alu102+11146);
  var val62 = data5_222720[alu134];
  var alu135 = (alu102+11147);
  var val63 = data5_222720[alu135];
  var alu136 = (alu102+22272);
  var val64 = data5_222720[alu136];
  var alu137 = (alu102+22273);
  var val65 = data5_222720[alu137];
  var alu138 = (alu102+22274);
  var val66 = data5_222720[alu138];
  var alu139 = (alu102+22275);
  var val67 = data5_222720[alu139];
  var alu140 = (alu102+22276);
  var val68 = data5_222720[alu140];
  var alu141 = (alu102+22277);
  var val69 = data5_222720[alu141];
  var alu142 = (alu102+22278);
  var val70 = data5_222720[alu142];
  var alu143 = (alu102+22279);
  var val71 = data5_222720[alu143];
  var alu144 = (alu102+22280);
  var val72 = data5_222720[alu144];
  var alu145 = (alu102+22281);
  var val73 = data5_222720[alu145];
  var alu146 = (alu102+22282);
  var val74 = data5_222720[alu146];
  var alu147 = (alu102+22283);
  var val75 = data5_222720[alu147];
  var alu148 = (alu102+33408);
  var val76 = data5_222720[alu148];
  var alu149 = (alu102+33409);
  var val77 = data5_222720[alu149];
  var alu150 = (alu102+33410);
  var val78 = data5_222720[alu150];
  var alu151 = (alu102+33411);
  var val79 = data5_222720[alu151];
  var alu152 = (alu102+33412);
  var val80 = data5_222720[alu152];
  var alu153 = (alu102+33413);
  var val81 = data5_222720[alu153];
  var alu154 = (alu102+33414);
  var val82 = data5_222720[alu154];
  var alu155 = (alu102+33415);
  var val83 = data5_222720[alu155];
  var alu156 = (alu102+33416);
  var val84 = data5_222720[alu156];
  var alu157 = (alu102+33417);
  var val85 = data5_222720[alu157];
  var alu158 = (alu102+33418);
  var val86 = data5_222720[alu158];
  var alu159 = (alu102+33419);
  var val87 = data5_222720[alu159];
  data0_222720[alu124] = (((acc0[1]+val17)*val19)+val52);
  data0_222720[alu125] = (((acc0[13]+val16)*val18)+val53);
  data0_222720[alu126] = (((acc0[25]+val22)*val26)+val54);
  data0_222720[alu127] = (((acc0[37]+val23)*val28)+val55);
  data0_222720[alu128] = (((acc0[5]+val24)*val30)+val56);
  data0_222720[alu129] = (((acc0[17]+val25)*val32)+val57);
  data0_222720[alu130] = (((acc0[29]+val34)*val35)+val58);
  data0_222720[alu131] = (((acc0[41]+val37)*val38)+val59);
  data0_222720[alu132] = (((acc0[9]+val40)*val41)+val60);
  data0_222720[alu133] = (((acc0[21]+val43)*val44)+val61);
  data0_222720[alu134] = (((acc0[33]+val46)*val47)+val62);
  data0_222720[alu135] = (((acc0[45]+val49)*val50)+val63);
  data0_222720[alu136] = (((acc0[2]+val17)*val19)+val64);
  data0_222720[alu137] = (((acc0[14]+val16)*val18)+val65);
  data0_222720[alu138] = (((acc0[26]+val22)*val26)+val66);
  data0_222720[alu139] = (((acc0[38]+val23)*val28)+val67);
  data0_222720[alu140] = (((acc0[6]+val24)*val30)+val68);
  data0_222720[alu141] = (((acc0[18]+val25)*val32)+val69);
  data0_222720[alu142] = (((acc0[30]+val34)*val35)+val70);
  data0_222720[alu143] = (((acc0[42]+val37)*val38)+val71);
  data0_222720[alu144] = (((acc0[10]+val40)*val41)+val72);
  data0_222720[alu145] = (((acc0[22]+val43)*val44)+val73);
  data0_222720[alu146] = (((acc0[34]+val46)*val47)+val74);
  data0_222720[alu147] = (((acc0[46]+val49)*val50)+val75);
  data0_222720[alu148] = (((acc0[3]+val17)*val19)+val76);
  data0_222720[alu149] = (((acc0[15]+val16)*val18)+val77);
  data0_222720[alu150] = (((acc0[27]+val22)*val26)+val78);
  data0_222720[alu151] = (((acc0[39]+val23)*val28)+val79);
  data0_222720[alu152] = (((acc0[7]+val24)*val30)+val80);
  data0_222720[alu153] = (((acc0[19]+val25)*val32)+val81);
  data0_222720[alu154] = (((acc0[31]+val34)*val35)+val82);
  data0_222720[alu155] = (((acc0[43]+val37)*val38)+val83);
  data0_222720[alu156] = (((acc0[11]+val40)*val41)+val84);
  data0_222720[alu157] = (((acc0[23]+val43)*val44)+val85);
  data0_222720[alu158] = (((acc0[35]+val46)*val47)+val86);
  data0_222720[alu159] = (((acc0[47]+val49)*val50)+val87);
  data0_222720[alu103] = (((acc0[12]+val16)*val18)+val20);
  data0_222720[alu108] = (((acc0[24]+val22)*val26)+val27);
  data0_222720[alu109] = (((acc0[36]+val23)*val28)+val29);
  data0_222720[alu110] = (((acc0[4]+val24)*val30)+val31);
  data0_222720[alu111] = (((acc0[16]+val25)*val32)+val33);
  data0_222720[alu113] = (((acc0[28]+val34)*val35)+val36);
  data0_222720[alu115] = (((acc0[40]+val37)*val38)+val39);
  data0_222720[alu117] = (((acc0[8]+val40)*val41)+val42);
  data0_222720[alu119] = (((acc0[20]+val43)*val44)+val45);
  data0_222720[alu121] = (((acc0[32]+val46)*val47)+val48);
  data0_222720[alu123] = (((acc0[44]+val49)*val50)+val51);
  data0_222720[alu102] = (((acc0[0]+val17)*val19)+val21);
}`;

const r_580_16_24n2 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32,16>;
@group(0) @binding(1)var<storage,read_write>data0_580:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_222720:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,1>;
  var acc1: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 580 */
  var lidx0 = i32(lindex.x); /* 16 */
  acc0[0] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 24; Ridx0++) {
    var val0 = data1_222720[((lidx0*24)+Ridx0+(gidx0*384))];
    acc0[0] = (acc0[0]+val0);
  }
  temp0[lidx0] = acc0[0];
  workgroupBarrier();
  acc1[0] = 0.0f;
  for (var Ridx102 = 0; Ridx102 < 16; Ridx102++) {
    var val1 = temp0[Ridx102];
    acc1[0] = (acc1[0]+val1);
  }
  var alu8 = (lidx0==0);
  if (alu8) {
    data0_580[gidx0] = (acc1[0]*0.0026041666666666665f);
  }
}`;

const r_580_16_24n3 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32,16>;
@group(0) @binding(1)var<storage,read_write>data0_580:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_222720:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_580:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,1>;
  var acc1: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 580 */
  var val0 = data2_580[gidx0];
  var lidx0 = i32(lindex.x); /* 16 */
  acc0[0] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 24; Ridx0++) {
    var val1 = data1_222720[((lidx0*24)+Ridx0+(gidx0*384))];
    var alu1 = (val1-val0);
    acc0[0] = (acc0[0]+(alu1*alu1));
  }
  temp0[lidx0] = acc0[0];
  workgroupBarrier();
  acc1[0] = 0.0f;
  for (var Ridx102 = 0; Ridx102 < 16; Ridx102++) {
    var val2 = temp0[Ridx102];
    acc1[0] = (acc1[0]+val2);
  }
  var alu9 = (lidx0==0);
  if (alu9) {
    data0_580[gidx0] = (1/sqrt(((acc1[0]*0.0026041666666666665f)+1e-05f)));
  }
}`;

const E_580_48_8 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_222720:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_222720:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_580:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_580:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_384:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_384:array<f32>;
@compute @workgroup_size(8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 48 */
  var gidx1 = i32(gindex.y); /* 580 */
  var lidx0 = i32(lindex.x); /* 8 */
  var alu0 = (lidx0+bitcast<i32>((bitcast<u32>(gidx0)<<3u)));
  var alu1 = (alu0+(gidx1*384));
  var val0 = data1_222720[alu1];
  var val1 = data2_580[gidx1];
  var val2 = data3_580[gidx1];
  var val3 = data4_384[alu0];
  var val4 = data5_384[alu0];
  data0_222720[alu1] = (((val0-val1)*val2*val3)+val4);
}`;

const E_12_2_12_2_32_12 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_221184:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_222720:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_580:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_580:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_384:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_384:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 24 */
  var gidx1 = i32(gindex.y); /* 2 */
  var gidx2 = i32(gindex.z); /* 12 */
  var lidx0 = i32(lindex.x); /* 32 */
  var alu0 = (lidx0+bitcast<i32>((bitcast<u32>(gidx2)<<5u)));
  var alu1 = (gidx0>>1u);
  var alu2 = (gidx0&1);
  var alu3 = (alu0+(alu1*4608)+(gidx1*111360)+(alu2*55680));
  var val0 = data1_222720[(alu3+384)];
  var alu4 = ((gidx1*290)+(alu2*145)+(alu1*12));
  var alu5 = (alu4+1);
  var val1 = data2_580[alu5];
  var val2 = data3_580[alu5];
  var val3 = data4_384[alu0];
  var val4 = data5_384[alu0];
  var val5 = data1_222720[(alu3+768)];
  var alu6 = (alu4+2);
  var val6 = data2_580[alu6];
  var alu7 = (alu4+3);
  var val7 = data2_580[alu7];
  var val8 = data3_580[alu6];
  var val9 = data1_222720[(alu3+1152)];
  var val10 = data3_580[alu7];
  var val11 = data1_222720[(alu3+1536)];
  var alu8 = (alu4+4);
  var val12 = data2_580[alu8];
  var alu9 = (alu4+5);
  var val13 = data2_580[alu9];
  var val14 = data3_580[alu8];
  var val15 = data1_222720[(alu3+1920)];
  var val16 = data3_580[alu9];
  var val17 = data1_222720[(alu3+2304)];
  var alu10 = (alu4+6);
  var val18 = data2_580[alu10];
  var alu11 = (alu4+8);
  var val19 = data2_580[alu11];
  var val20 = data3_580[alu10];
  var val21 = data1_222720[(alu3+2688)];
  var alu12 = (alu4+7);
  var val22 = data2_580[alu12];
  var val23 = data3_580[alu12];
  var val24 = data1_222720[(alu3+3072)];
  var val25 = data3_580[alu11];
  var val26 = data1_222720[(alu3+3456)];
  var alu13 = (alu4+9);
  var val27 = data2_580[alu13];
  var val28 = data3_580[alu13];
  var val29 = data1_222720[(alu3+3840)];
  var alu14 = (alu4+10);
  var val30 = data2_580[alu14];
  var alu15 = (alu4+11);
  var val31 = data2_580[alu15];
  var val32 = data3_580[alu14];
  var val33 = data1_222720[(alu3+4224)];
  var val34 = data3_580[alu15];
  var val35 = data1_222720[(alu3+4608)];
  var alu16 = (alu4+12);
  var val36 = data2_580[alu16];
  var val37 = data3_580[alu16];
  var alu17 = ((gidx0*12)+(gidx1*288)+(gidx2*18432)+(lidx0*576));
  data0_221184[(alu17+1)] = (((val5-val6)*val8*val3)+val4);
  data0_221184[(alu17+2)] = (((val9-val7)*val10*val3)+val4);
  data0_221184[(alu17+3)] = (((val11-val12)*val14*val3)+val4);
  data0_221184[(alu17+4)] = (((val15-val13)*val16*val3)+val4);
  data0_221184[(alu17+5)] = (((val17-val18)*val20*val3)+val4);
  data0_221184[(alu17+6)] = (((val21-val22)*val23*val3)+val4);
  data0_221184[(alu17+7)] = (((val24-val19)*val25*val3)+val4);
  data0_221184[(alu17+8)] = (((val26-val27)*val28*val3)+val4);
  data0_221184[(alu17+9)] = (((val29-val30)*val32*val3)+val4);
  data0_221184[(alu17+10)] = (((val33-val31)*val34*val3)+val4);
  data0_221184[(alu17+11)] = (((val35-val36)*val37*val3)+val4);
  data0_221184[alu17] = (((val0-val1)*val2*val3)+val4);
}`;

const r_20_32_29_4_3_384 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_222720:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_222720:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_147456:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_384:array<f32>;
@compute @workgroup_size(29) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,12>;
  var gidx0 = i32(gindex.x); /* 32 */
  var gidx1 = i32(gindex.y); /* 20 */
  var lidx0 = i32(lindex.x); /* 29 */
  var alu0 = ((gidx1*11136)+(lidx0*384));
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  acc0[4] = 0.0f;
  acc0[5] = 0.0f;
  acc0[6] = 0.0f;
  acc0[7] = 0.0f;
  acc0[8] = 0.0f;
  acc0[9] = 0.0f;
  acc0[10] = 0.0f;
  acc0[11] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 384; Ridx0++) {
    var val0 = data1_222720[(alu0+Ridx0)];
    var alu13 = ((gidx0*4608)+Ridx0);
    var val1 = data2_147456[alu13];
    var val2 = data2_147456[(alu13+1536)];
    var val3 = data2_147456[(alu13+384)];
    var val4 = data2_147456[(alu13+3072)];
    var val5 = data2_147456[(alu13+768)];
    var val6 = data2_147456[(alu13+1920)];
    var val7 = data2_147456[(alu13+3456)];
    var val8 = data2_147456[(alu13+2304)];
    var val9 = data2_147456[(alu13+3840)];
    var val10 = data2_147456[(alu13+1152)];
    var val11 = data2_147456[(alu13+2688)];
    var val12 = data2_147456[(alu13+4224)];
    acc0[0] = (acc0[0]+(val0*val1));
    acc0[1] = (acc0[1]+(val0*val2));
    acc0[2] = (acc0[2]+(val0*val4));
    acc0[3] = (acc0[3]+(val0*val3));
    acc0[4] = (acc0[4]+(val0*val6));
    acc0[5] = (acc0[5]+(val0*val7));
    acc0[6] = (acc0[6]+(val0*val5));
    acc0[7] = (acc0[7]+(val0*val8));
    acc0[8] = (acc0[8]+(val0*val9));
    acc0[9] = (acc0[9]+(val0*val10));
    acc0[10] = (acc0[10]+(val0*val11));
    acc0[11] = (acc0[11]+(val0*val12));
  }
  var alu27 = (gidx0*12);
  var val13 = data3_384[(alu27+4)];
  var val14 = data3_384[alu27];
  var val15 = data3_384[(alu27+1)];
  var val16 = data3_384[(alu27+2)];
  var val17 = data3_384[(alu27+3)];
  var val18 = data3_384[(alu27+5)];
  var val19 = data3_384[(alu27+6)];
  var val20 = data3_384[(alu27+7)];
  var val21 = data3_384[(alu27+8)];
  var val22 = data3_384[(alu27+9)];
  var val23 = data3_384[(alu27+10)];
  var val24 = data3_384[(alu27+11)];
  var alu28 = (alu0+alu27);
  data0_222720[(alu28+1)] = (acc0[3]+val15);
  data0_222720[(alu28+2)] = (acc0[6]+val16);
  data0_222720[(alu28+3)] = (acc0[9]+val17);
  data0_222720[(alu28+4)] = (acc0[1]+val13);
  data0_222720[(alu28+5)] = (acc0[4]+val18);
  data0_222720[(alu28+6)] = (acc0[7]+val19);
  data0_222720[(alu28+7)] = (acc0[10]+val20);
  data0_222720[(alu28+8)] = (acc0[2]+val21);
  data0_222720[(alu28+9)] = (acc0[5]+val22);
  data0_222720[(alu28+10)] = (acc0[8]+val23);
  data0_222720[(alu28+11)] = (acc0[11]+val24);
  data0_222720[alu28] = (acc0[0]+val14);
}`;

const r_6_116_20_29_5_64 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_2018400:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_222720:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_222720:array<f32>;
@compute @workgroup_size(29) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,5>;
  var gidx0 = i32(gindex.x); /* 20 */
  var gidx1 = i32(gindex.y); /* 116 */
  var gidx2 = i32(gindex.z); /* 6 */
  var lidx0 = i32(lindex.x); /* 29 */
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  acc0[4] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 64; Ridx0++) {
    var alu5 = (bitcast<i32>((bitcast<u32>(gidx2)<<6u))+Ridx0);
    var val0 = data1_222720[((gidx0*11136)+(lidx0*384)+alu5)];
    var alu6 = (alu5+(gidx1*1920));
    var val1 = data2_222720[alu6];
    var val2 = data2_222720[(alu6+384)];
    var val3 = data2_222720[(alu6+768)];
    var val4 = data2_222720[(alu6+1152)];
    var val5 = data2_222720[(alu6+1536)];
    acc0[0] = (acc0[0]+(val0*val1));
    acc0[1] = (acc0[1]+(val0*val2));
    acc0[2] = (acc0[2]+(val0*val3));
    acc0[3] = (acc0[3]+(val0*val4));
    acc0[4] = (acc0[4]+(val0*val5));
  }
  var alu13 = ((gidx0*16820)+(lidx0*580)+(gidx1*5)+(gidx2*336400));
  data0_2018400[(alu13+1)] = (acc0[1]*0.125f);
  data0_2018400[(alu13+2)] = (acc0[2]*0.125f);
  data0_2018400[(alu13+3)] = (acc0[3]*0.125f);
  data0_2018400[(alu13+4)] = (acc0[4]*0.125f);
  data0_2018400[alu13] = (acc0[0]*0.125f);
}`;

const r_870_4_580 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_3480:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_2018400:array<f32>;
@compute @workgroup_size(4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 870 */
  var lidx0 = i32(lindex.x); /* 4 */
  acc0[0] = (f32(-INFINITY));
  for (var Ridx0 = 0; Ridx0 < 580; Ridx0++) {
    var val0 = data1_2018400[((gidx0*2320)+(lidx0*580)+Ridx0)];
    var alu1 = select(acc0[0],val0,(acc0[0]<val0));
    acc0[0] = alu1;
  }
  data0_3480[(lidx0+bitcast<i32>((bitcast<u32>(gidx0)<<2u)))] = acc0[0];
}`;

const r_435_8_580 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_3480:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_2018400:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_3480:array<f32>;
@compute @workgroup_size(8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 435 */
  var lidx0 = i32(lindex.x); /* 8 */
  var alu0 = (lidx0+bitcast<i32>((bitcast<u32>(gidx0)<<3u)));
  var val0 = data2_3480[alu0];
  acc0[0] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 580; Ridx0++) {
    var val1 = data1_2018400[((gidx0*4640)+(lidx0*580)+Ridx0)];
    acc0[0] = (acc0[0]+exp2(((val1-val0)*1.4426950408889634f)));
  }
  data0_3480[alu0] = acc0[0];
}`;

const r_116_3_4_16_2_5_580 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_222720:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_2018400:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_3480:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_3480:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_222720:array<f32>;
@compute @workgroup_size(16,2) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,5>;
  var gidx1 = i32(gindex.y); /* 3 */
  var gidx2 = i32(gindex.z); /* 116 */
  var lidx1 = i32(lindex.y); /* 2 */
  var alu0 = ((gidx1*1160)+(lidx1*580)+(gidx2*5));
  var alu1 = (alu0+1);
  var val0 = data2_3480[alu1];
  var alu2 = (alu0+2);
  var val1 = data2_3480[alu2];
  var alu3 = (alu0+3);
  var val2 = data2_3480[alu3];
  var alu4 = (alu0+4);
  var val3 = data2_3480[alu4];
  var val4 = data2_3480[alu0];
  var gidx0 = i32(gindex.x); /* 4 */
  var lidx0 = i32(lindex.x); /* 16 */
  var alu5 = (lidx0+bitcast<i32>((bitcast<u32>(gidx0)<<4u))+bitcast<i32>((bitcast<u32>(gidx1)<<7u))+bitcast<i32>((bitcast<u32>(lidx1)<<6u)));
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  acc0[4] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 580; Ridx0++) {
    var alu11 = ((gidx1*672800)+(lidx1*336400)+(gidx2*2900)+Ridx0);
    var val5 = data1_2018400[alu11];
    var val6 = data4_222720[(alu5+(Ridx0*384))];
    var val7 = data1_2018400[(alu11+580)];
    var val8 = data1_2018400[(alu11+1160)];
    var val9 = data1_2018400[(alu11+1740)];
    var val10 = data1_2018400[(alu11+2320)];
    acc0[0] = (acc0[0]+(exp2(((val5-val4)*1.4426950408889634f))*val6));
    acc0[1] = (acc0[1]+(exp2(((val7-val0)*1.4426950408889634f))*val6));
    acc0[2] = (acc0[2]+(exp2(((val8-val1)*1.4426950408889634f))*val6));
    acc0[3] = (acc0[3]+(exp2(((val9-val2)*1.4426950408889634f))*val6));
    acc0[4] = (acc0[4]+(exp2(((val10-val3)*1.4426950408889634f))*val6));
  }
  var val11 = data3_3480[alu0];
  var val12 = data3_3480[alu1];
  var val13 = data3_3480[alu2];
  var val14 = data3_3480[alu3];
  var val15 = data3_3480[alu4];
  var alu18 = (alu5+(gidx2*1920));
  data0_222720[alu18] = (acc0[0]*(1/val11));
  data0_222720[(alu18+384)] = (acc0[1]*(1/val12));
  data0_222720[(alu18+768)] = (acc0[2]*(1/val13));
  data0_222720[(alu18+1152)] = (acc0[3]*(1/val14));
  data0_222720[(alu18+1536)] = (acc0[4]*(1/val15));
}`;

const r_5_32_29_4_3_4_384n1 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_222720:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_222720:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_147456:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_384:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_384:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_222720:array<f32>;
@compute @workgroup_size(29) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,48>;
  var gidx0 = i32(gindex.x); /* 32 */
  var gidx1 = i32(gindex.y); /* 5 */
  var lidx0 = i32(lindex.x); /* 29 */
  var alu0 = ((gidx1*44544)+(lidx0*384));
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  acc0[4] = 0.0f;
  acc0[5] = 0.0f;
  acc0[6] = 0.0f;
  acc0[7] = 0.0f;
  acc0[8] = 0.0f;
  acc0[9] = 0.0f;
  acc0[10] = 0.0f;
  acc0[11] = 0.0f;
  acc0[12] = 0.0f;
  acc0[13] = 0.0f;
  acc0[14] = 0.0f;
  acc0[15] = 0.0f;
  acc0[16] = 0.0f;
  acc0[17] = 0.0f;
  acc0[18] = 0.0f;
  acc0[19] = 0.0f;
  acc0[20] = 0.0f;
  acc0[21] = 0.0f;
  acc0[22] = 0.0f;
  acc0[23] = 0.0f;
  acc0[24] = 0.0f;
  acc0[25] = 0.0f;
  acc0[26] = 0.0f;
  acc0[27] = 0.0f;
  acc0[28] = 0.0f;
  acc0[29] = 0.0f;
  acc0[30] = 0.0f;
  acc0[31] = 0.0f;
  acc0[32] = 0.0f;
  acc0[33] = 0.0f;
  acc0[34] = 0.0f;
  acc0[35] = 0.0f;
  acc0[36] = 0.0f;
  acc0[37] = 0.0f;
  acc0[38] = 0.0f;
  acc0[39] = 0.0f;
  acc0[40] = 0.0f;
  acc0[41] = 0.0f;
  acc0[42] = 0.0f;
  acc0[43] = 0.0f;
  acc0[44] = 0.0f;
  acc0[45] = 0.0f;
  acc0[46] = 0.0f;
  acc0[47] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 384; Ridx0++) {
    var alu49 = (alu0+Ridx0);
    var val0 = data1_222720[alu49];
    var alu50 = ((gidx0*4608)+Ridx0);
    var val1 = data2_147456[alu50];
    var val2 = data1_222720[(alu49+11136)];
    var val3 = data1_222720[(alu49+22272)];
    var val4 = data1_222720[(alu49+33408)];
    var val5 = data2_147456[(alu50+1536)];
    var val6 = data2_147456[(alu50+3072)];
    var val7 = data2_147456[(alu50+384)];
    var val8 = data2_147456[(alu50+1920)];
    var val9 = data2_147456[(alu50+3456)];
    var val10 = data2_147456[(alu50+768)];
    var val11 = data2_147456[(alu50+2304)];
    var val12 = data2_147456[(alu50+3840)];
    var val13 = data2_147456[(alu50+1152)];
    var val14 = data2_147456[(alu50+2688)];
    var val15 = data2_147456[(alu50+4224)];
    acc0[0] = (acc0[0]+(val0*val1));
    acc0[1] = (acc0[1]+(val2*val1));
    acc0[2] = (acc0[2]+(val3*val1));
    acc0[3] = (acc0[3]+(val4*val1));
    acc0[4] = (acc0[4]+(val0*val5));
    acc0[5] = (acc0[5]+(val2*val5));
    acc0[6] = (acc0[6]+(val3*val5));
    acc0[7] = (acc0[7]+(val4*val5));
    acc0[8] = (acc0[8]+(val0*val6));
    acc0[9] = (acc0[9]+(val2*val6));
    acc0[10] = (acc0[10]+(val3*val6));
    acc0[11] = (acc0[11]+(val4*val6));
    acc0[12] = (acc0[12]+(val0*val7));
    acc0[13] = (acc0[13]+(val2*val7));
    acc0[14] = (acc0[14]+(val3*val7));
    acc0[15] = (acc0[15]+(val4*val7));
    acc0[16] = (acc0[16]+(val0*val8));
    acc0[17] = (acc0[17]+(val2*val8));
    acc0[18] = (acc0[18]+(val3*val8));
    acc0[19] = (acc0[19]+(val4*val8));
    acc0[20] = (acc0[20]+(val0*val9));
    acc0[21] = (acc0[21]+(val2*val9));
    acc0[22] = (acc0[22]+(val3*val9));
    acc0[23] = (acc0[23]+(val4*val9));
    acc0[24] = (acc0[24]+(val0*val10));
    acc0[25] = (acc0[25]+(val2*val10));
    acc0[26] = (acc0[26]+(val3*val10));
    acc0[27] = (acc0[27]+(val4*val10));
    acc0[28] = (acc0[28]+(val0*val11));
    acc0[29] = (acc0[29]+(val2*val11));
    acc0[30] = (acc0[30]+(val3*val11));
    acc0[31] = (acc0[31]+(val4*val11));
    acc0[32] = (acc0[32]+(val0*val12));
    acc0[33] = (acc0[33]+(val2*val12));
    acc0[34] = (acc0[34]+(val3*val12));
    acc0[35] = (acc0[35]+(val4*val12));
    acc0[36] = (acc0[36]+(val0*val13));
    acc0[37] = (acc0[37]+(val2*val13));
    acc0[38] = (acc0[38]+(val3*val13));
    acc0[39] = (acc0[39]+(val4*val13));
    acc0[40] = (acc0[40]+(val0*val14));
    acc0[41] = (acc0[41]+(val2*val14));
    acc0[42] = (acc0[42]+(val3*val14));
    acc0[43] = (acc0[43]+(val4*val14));
    acc0[44] = (acc0[44]+(val0*val15));
    acc0[45] = (acc0[45]+(val2*val15));
    acc0[46] = (acc0[46]+(val3*val15));
    acc0[47] = (acc0[47]+(val4*val15));
  }
  var alu100 = (gidx0*12);
  var alu101 = (alu100+1);
  var val16 = data3_384[alu101];
  var val17 = data3_384[alu100];
  var val18 = data4_384[alu101];
  var val19 = data4_384[alu100];
  var alu102 = (alu0+alu100);
  var alu103 = (alu102+1);
  var val20 = data5_222720[alu103];
  var val21 = data5_222720[alu102];
  var alu104 = (alu100+2);
  var val22 = data3_384[alu104];
  var alu105 = (alu100+3);
  var val23 = data3_384[alu105];
  var alu106 = (alu100+4);
  var val24 = data3_384[alu106];
  var alu107 = (alu100+5);
  var val25 = data3_384[alu107];
  var val26 = data4_384[alu104];
  var alu108 = (alu102+2);
  var val27 = data5_222720[alu108];
  var val28 = data4_384[alu105];
  var alu109 = (alu102+3);
  var val29 = data5_222720[alu109];
  var val30 = data4_384[alu106];
  var alu110 = (alu102+4);
  var val31 = data5_222720[alu110];
  var val32 = data4_384[alu107];
  var alu111 = (alu102+5);
  var val33 = data5_222720[alu111];
  var alu112 = (alu100+6);
  var val34 = data3_384[alu112];
  var val35 = data4_384[alu112];
  var alu113 = (alu102+6);
  var val36 = data5_222720[alu113];
  var alu114 = (alu100+7);
  var val37 = data3_384[alu114];
  var val38 = data4_384[alu114];
  var alu115 = (alu102+7);
  var val39 = data5_222720[alu115];
  var alu116 = (alu100+8);
  var val40 = data3_384[alu116];
  var val41 = data4_384[alu116];
  var alu117 = (alu102+8);
  var val42 = data5_222720[alu117];
  var alu118 = (alu100+9);
  var val43 = data3_384[alu118];
  var val44 = data4_384[alu118];
  var alu119 = (alu102+9);
  var val45 = data5_222720[alu119];
  var alu120 = (alu100+10);
  var val46 = data3_384[alu120];
  var val47 = data4_384[alu120];
  var alu121 = (alu102+10);
  var val48 = data5_222720[alu121];
  var alu122 = (alu100+11);
  var val49 = data3_384[alu122];
  var val50 = data4_384[alu122];
  var alu123 = (alu102+11);
  var val51 = data5_222720[alu123];
  var alu124 = (alu102+11136);
  var val52 = data5_222720[alu124];
  var alu125 = (alu102+11137);
  var val53 = data5_222720[alu125];
  var alu126 = (alu102+11138);
  var val54 = data5_222720[alu126];
  var alu127 = (alu102+11139);
  var val55 = data5_222720[alu127];
  var alu128 = (alu102+11140);
  var val56 = data5_222720[alu128];
  var alu129 = (alu102+11141);
  var val57 = data5_222720[alu129];
  var alu130 = (alu102+11142);
  var val58 = data5_222720[alu130];
  var alu131 = (alu102+11143);
  var val59 = data5_222720[alu131];
  var alu132 = (alu102+11144);
  var val60 = data5_222720[alu132];
  var alu133 = (alu102+11145);
  var val61 = data5_222720[alu133];
  var alu134 = (alu102+11146);
  var val62 = data5_222720[alu134];
  var alu135 = (alu102+11147);
  var val63 = data5_222720[alu135];
  var alu136 = (alu102+22272);
  var val64 = data5_222720[alu136];
  var alu137 = (alu102+22273);
  var val65 = data5_222720[alu137];
  var alu138 = (alu102+22274);
  var val66 = data5_222720[alu138];
  var alu139 = (alu102+22275);
  var val67 = data5_222720[alu139];
  var alu140 = (alu102+22276);
  var val68 = data5_222720[alu140];
  var alu141 = (alu102+22277);
  var val69 = data5_222720[alu141];
  var alu142 = (alu102+22278);
  var val70 = data5_222720[alu142];
  var alu143 = (alu102+22279);
  var val71 = data5_222720[alu143];
  var alu144 = (alu102+22280);
  var val72 = data5_222720[alu144];
  var alu145 = (alu102+22281);
  var val73 = data5_222720[alu145];
  var alu146 = (alu102+22282);
  var val74 = data5_222720[alu146];
  var alu147 = (alu102+22283);
  var val75 = data5_222720[alu147];
  var alu148 = (alu102+33408);
  var val76 = data5_222720[alu148];
  var alu149 = (alu102+33409);
  var val77 = data5_222720[alu149];
  var alu150 = (alu102+33410);
  var val78 = data5_222720[alu150];
  var alu151 = (alu102+33411);
  var val79 = data5_222720[alu151];
  var alu152 = (alu102+33412);
  var val80 = data5_222720[alu152];
  var alu153 = (alu102+33413);
  var val81 = data5_222720[alu153];
  var alu154 = (alu102+33414);
  var val82 = data5_222720[alu154];
  var alu155 = (alu102+33415);
  var val83 = data5_222720[alu155];
  var alu156 = (alu102+33416);
  var val84 = data5_222720[alu156];
  var alu157 = (alu102+33417);
  var val85 = data5_222720[alu157];
  var alu158 = (alu102+33418);
  var val86 = data5_222720[alu158];
  var alu159 = (alu102+33419);
  var val87 = data5_222720[alu159];
  data0_222720[alu124] = (((acc0[1]+val17)*val19)+val52);
  data0_222720[alu125] = (((acc0[13]+val16)*val18)+val53);
  data0_222720[alu126] = (((acc0[25]+val22)*val26)+val54);
  data0_222720[alu127] = (((acc0[37]+val23)*val28)+val55);
  data0_222720[alu128] = (((acc0[5]+val24)*val30)+val56);
  data0_222720[alu129] = (((acc0[17]+val25)*val32)+val57);
  data0_222720[alu130] = (((acc0[29]+val34)*val35)+val58);
  data0_222720[alu131] = (((acc0[41]+val37)*val38)+val59);
  data0_222720[alu132] = (((acc0[9]+val40)*val41)+val60);
  data0_222720[alu133] = (((acc0[21]+val43)*val44)+val61);
  data0_222720[alu134] = (((acc0[33]+val46)*val47)+val62);
  data0_222720[alu135] = (((acc0[45]+val49)*val50)+val63);
  data0_222720[alu136] = (((acc0[2]+val17)*val19)+val64);
  data0_222720[alu137] = (((acc0[14]+val16)*val18)+val65);
  data0_222720[alu138] = (((acc0[26]+val22)*val26)+val66);
  data0_222720[alu139] = (((acc0[38]+val23)*val28)+val67);
  data0_222720[alu140] = (((acc0[6]+val24)*val30)+val68);
  data0_222720[alu141] = (((acc0[18]+val25)*val32)+val69);
  data0_222720[alu142] = (((acc0[30]+val34)*val35)+val70);
  data0_222720[alu143] = (((acc0[42]+val37)*val38)+val71);
  data0_222720[alu144] = (((acc0[10]+val40)*val41)+val72);
  data0_222720[alu145] = (((acc0[22]+val43)*val44)+val73);
  data0_222720[alu146] = (((acc0[34]+val46)*val47)+val74);
  data0_222720[alu147] = (((acc0[46]+val49)*val50)+val75);
  data0_222720[alu148] = (((acc0[3]+val17)*val19)+val76);
  data0_222720[alu149] = (((acc0[15]+val16)*val18)+val77);
  data0_222720[alu150] = (((acc0[27]+val22)*val26)+val78);
  data0_222720[alu151] = (((acc0[39]+val23)*val28)+val79);
  data0_222720[alu152] = (((acc0[7]+val24)*val30)+val80);
  data0_222720[alu153] = (((acc0[19]+val25)*val32)+val81);
  data0_222720[alu154] = (((acc0[31]+val34)*val35)+val82);
  data0_222720[alu155] = (((acc0[43]+val37)*val38)+val83);
  data0_222720[alu156] = (((acc0[11]+val40)*val41)+val84);
  data0_222720[alu157] = (((acc0[23]+val43)*val44)+val85);
  data0_222720[alu158] = (((acc0[35]+val46)*val47)+val86);
  data0_222720[alu159] = (((acc0[47]+val49)*val50)+val87);
  data0_222720[alu103] = (((acc0[12]+val16)*val18)+val20);
  data0_222720[alu108] = (((acc0[24]+val22)*val26)+val27);
  data0_222720[alu109] = (((acc0[36]+val23)*val28)+val29);
  data0_222720[alu110] = (((acc0[4]+val24)*val30)+val31);
  data0_222720[alu111] = (((acc0[16]+val25)*val32)+val33);
  data0_222720[alu113] = (((acc0[28]+val34)*val35)+val36);
  data0_222720[alu115] = (((acc0[40]+val37)*val38)+val39);
  data0_222720[alu117] = (((acc0[8]+val40)*val41)+val42);
  data0_222720[alu119] = (((acc0[20]+val43)*val44)+val45);
  data0_222720[alu121] = (((acc0[32]+val46)*val47)+val48);
  data0_222720[alu123] = (((acc0[44]+val49)*val50)+val51);
  data0_222720[alu102] = (((acc0[0]+val17)*val19)+val21);
}`;

const E_9216_32_3 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_884736:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_221184:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_221184:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_221184:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_221184:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 9216 */
  var lidx0 = i32(lindex.x); /* 32 */
  var alu0 = (lidx0+(gidx0*96));
  var alu1 = (gidx0<2304);
  var val0 = select(0.0f, data1_221184[alu0], alu1);
  var alu2 = (alu0+32);
  var val1 = select(0.0f, data1_221184[alu2], alu1);
  var alu3 = (alu0+64);
  var val2 = select(0.0f, data1_221184[alu3], alu1);
  var alu4 = ((2303<gidx0)&(gidx0<4608));
  var val3 = select(0.0f, data2_221184[(alu0+-221184)], alu4);
  var val4 = select(0.0f, data2_221184[(alu0+-221152)], alu4);
  var val5 = select(0.0f, data2_221184[(alu0+-221120)], alu4);
  var alu5 = ((4607<gidx0)&(gidx0<6912));
  var val6 = select(0.0f, data3_221184[(alu0+-442368)], alu5);
  var val7 = select(0.0f, data3_221184[(alu0+-442336)], alu5);
  var val8 = select(0.0f, data3_221184[(alu0+-442304)], alu5);
  var alu6 = (6911<gidx0);
  var val9 = select(0.0f, data4_221184[(alu0+-663552)], alu6);
  var val10 = select(0.0f, data4_221184[(alu0+-663520)], alu6);
  var val11 = select(0.0f, data4_221184[(alu0+-663488)], alu6);
  data0_884736[alu0] = (val0+val3+val6+val9);
  data0_884736[alu2] = (val1+val4+val7+val10);
  data0_884736[alu3] = (val2+val5+val8+val11);
}`;

const r_6_64_32_4_3_1536 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_147456:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_884736:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_393216:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,12>;
  var gidx0 = i32(gindex.x); /* 64 */
  var gidx1 = i32(gindex.y); /* 6 */
  var lidx0 = i32(lindex.x); /* 32 */
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  acc0[4] = 0.0f;
  acc0[5] = 0.0f;
  acc0[6] = 0.0f;
  acc0[7] = 0.0f;
  acc0[8] = 0.0f;
  acc0[9] = 0.0f;
  acc0[10] = 0.0f;
  acc0[11] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 1536; Ridx0++) {
    var alu12 = (lidx0+(gidx1*96)+(Ridx0*576));
    var val0 = data1_884736[alu12];
    var alu13 = ((gidx0*6144)+Ridx0);
    var val1 = data2_393216[alu13];
    var val2 = data1_884736[(alu12+32)];
    var val3 = data1_884736[(alu12+64)];
    var val4 = data2_393216[(alu13+1536)];
    var val5 = data2_393216[(alu13+3072)];
    var val6 = data2_393216[(alu13+4608)];
    acc0[0] = (acc0[0]+(val0*val1));
    acc0[1] = (acc0[1]+(val2*val1));
    acc0[2] = (acc0[2]+(val3*val1));
    acc0[3] = (acc0[3]+(val0*val4));
    acc0[4] = (acc0[4]+(val2*val4));
    acc0[5] = (acc0[5]+(val3*val4));
    acc0[6] = (acc0[6]+(val0*val5));
    acc0[7] = (acc0[7]+(val2*val5));
    acc0[8] = (acc0[8]+(val3*val5));
    acc0[9] = (acc0[9]+(val0*val6));
    acc0[10] = (acc0[10]+(val2*val6));
    acc0[11] = (acc0[11]+(val3*val6));
  }
  var alu27 = ((gidx1*24576)+bitcast<i32>((bitcast<u32>(lidx0)<<8u))+bitcast<i32>((bitcast<u32>(gidx0)<<2u)));
  data0_147456[(alu27+8192)] = acc0[1];
  data0_147456[(alu27+8193)] = acc0[4];
  data0_147456[(alu27+8194)] = acc0[7];
  data0_147456[(alu27+8195)] = acc0[10];
  data0_147456[(alu27+16384)] = acc0[2];
  data0_147456[(alu27+16385)] = acc0[5];
  data0_147456[(alu27+16386)] = acc0[8];
  data0_147456[(alu27+16387)] = acc0[11];
  data0_147456[(alu27+1)] = acc0[3];
  data0_147456[(alu27+2)] = acc0[6];
  data0_147456[(alu27+3)] = acc0[9];
  data0_147456[alu27] = acc0[0];
}`;

const r_576_16_16 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32,16>;
@group(0) @binding(1)var<storage,read_write>data0_576:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_147456:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,1>;
  var acc1: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 576 */
  var lidx0 = i32(lindex.x); /* 16 */
  acc0[0] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 16; Ridx0++) {
    var val0 = data1_147456[(bitcast<i32>((bitcast<u32>(lidx0)<<4u))+Ridx0+bitcast<i32>((bitcast<u32>(gidx0)<<8u)))];
    acc0[0] = (acc0[0]+val0);
  }
  temp0[lidx0] = acc0[0];
  workgroupBarrier();
  acc1[0] = 0.0f;
  for (var Ridx102 = 0; Ridx102 < 16; Ridx102++) {
    var val1 = temp0[Ridx102];
    acc1[0] = (acc1[0]+val1);
  }
  var alu8 = (lidx0==0);
  if (alu8) {
    data0_576[gidx0] = (acc1[0]*0.00390625f);
  }
}`;

const E_576_32_8 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_147456:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_147456:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_576:array<f32>;
@compute @workgroup_size(8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 32 */
  var gidx1 = i32(gindex.y); /* 576 */
  var lidx0 = i32(lindex.x); /* 8 */
  var alu0 = (lidx0+bitcast<i32>((bitcast<u32>(gidx0)<<3u))+bitcast<i32>((bitcast<u32>(gidx1)<<8u)));
  var val0 = data1_147456[alu0];
  var val1 = data2_576[gidx1];
  data0_147456[alu0] = (val0-val1);
}`;

const r_576_16_16n1 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32,16>;
@group(0) @binding(1)var<storage,read_write>data0_576:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_147456:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,1>;
  var acc1: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 576 */
  var lidx0 = i32(lindex.x); /* 16 */
  acc0[0] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 16; Ridx0++) {
    var val0 = data1_147456[(bitcast<i32>((bitcast<u32>(lidx0)<<4u))+Ridx0+bitcast<i32>((bitcast<u32>(gidx0)<<8u)))];
    acc0[0] = (acc0[0]+(val0*val0));
  }
  temp0[lidx0] = acc0[0];
  workgroupBarrier();
  acc1[0] = 0.0f;
  for (var Ridx102 = 0; Ridx102 < 16; Ridx102++) {
    var val1 = temp0[Ridx102];
    acc1[0] = (acc1[0]+val1);
  }
  var alu8 = (lidx0==0);
  if (alu8) {
    data0_576[gidx0] = sqrt(((acc1[0]*0.00390625f)+1e-06f));
  }
}`;

const E_32_192_8_3 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_147456:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_147456:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_576:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_256:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_256:array<f32>;
@compute @workgroup_size(8,3) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 192 */
  var gidx1 = i32(gindex.y); /* 32 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 3 */
  var alu0 = (lidx0+bitcast<i32>((bitcast<u32>(gidx1)<<3u)));
  var val0 = data1_147456[(alu0+(gidx0*768)+bitcast<i32>((bitcast<u32>(lidx1)<<8u)))];
  var alu1 = (lidx1+(gidx0*3));
  var val1 = data2_576[alu1];
  var val2 = data3_256[alu0];
  var val3 = data4_256[alu0];
  var alu2 = ((val0*(1/val1)*val2)+val3);
  data0_147456[(alu1+(gidx1*4608)+(lidx0*576))] = (alu2*(1/(1.0f+exp2((alu2*-1.4426950408889634f)))));
}`;

const r_6_24_8_16_4_128_3_3 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_73728:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_147456:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_147456:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 8 */
  var gidx1 = i32(gindex.y); /* 24 */
  var gidx2 = i32(gindex.z); /* 6 */
  var lidx0 = i32(lindex.x); /* 16 */
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 128; Ridx0++) {
    for (var Ridx1 = 0; Ridx1 < 3; Ridx1++) {
      for (var Ridx2 = 0; Ridx2 < 3; Ridx2++) {
        var alu4 = (gidx1+Ridx2);
        var alu5 = (alu4+(gidx2*96)+(Ridx1*24)+(Ridx0*576));
        var alu6 = ((0<alu4)&(alu4<25));
        var val0 = select(0.0f, data1_147456[(alu5+73703)], (alu6&(0<(gidx2+Ridx1))));
        var val1 = select(0.0f, data1_147456[(alu5+73727)], alu6);
        var val2 = select(0.0f, data1_147456[(alu5+73751)], alu6);
        var val3 = data2_147456[((Ridx1*3)+Ridx2+(Ridx0*9)+(gidx0*18432)+(lidx0*1152))];
        var val4 = select(0.0f, data1_147456[(alu5+73775)], (alu6&((bitcast<i32>((bitcast<u32>(gidx2)<<2u))+Ridx1)<22)));
        acc0[0] = (acc0[0]+(val0*val3));
        acc0[1] = (acc0[1]+(val1*val3));
        acc0[2] = (acc0[2]+(val2*val3));
        acc0[3] = (acc0[3]+(val4*val3));
      }
    }
  }
  var alu14 = (lidx0+bitcast<i32>((bitcast<u32>(gidx0)<<4u))+bitcast<i32>((bitcast<u32>(gidx1)<<7u))+(gidx2*12288));
  data0_73728[alu14] = acc0[0];
  data0_73728[(alu14+3072)] = acc0[1];
  data0_73728[(alu14+6144)] = acc0[2];
  data0_73728[(alu14+9216)] = acc0[3];
}`;

const r_144_4_128 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_576:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_73728:array<f32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 144 */
  var cast0 = bitcast<u32>(gidx0);
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 128; Ridx0++) {
    var alu4 = (bitcast<i32>((cast0<<9u))+Ridx0);
    var val0 = data1_73728[alu4];
    var val1 = data1_73728[(alu4+128)];
    var val2 = data1_73728[(alu4+256)];
    var val3 = data1_73728[(alu4+384)];
    acc0[0] = (acc0[0]+val0);
    acc0[1] = (acc0[1]+val1);
    acc0[2] = (acc0[2]+val2);
    acc0[3] = (acc0[3]+val3);
  }
  var cast1 = bitcast<i32>((cast0<<2u));
  data0_576[cast1] = (acc0[0]*0.0078125f);
  data0_576[(cast1+1)] = (acc0[1]*0.0078125f);
  data0_576[(cast1+2)] = (acc0[2]*0.0078125f);
  data0_576[(cast1+3)] = (acc0[3]*0.0078125f);
}`;

const E_72_128_8 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_73728:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_73728:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_576:array<f32>;
@compute @workgroup_size(8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 128 */
  var gidx1 = i32(gindex.y); /* 72 */
  var lidx0 = i32(lindex.x); /* 8 */
  var cast0 = bitcast<u32>(gidx1);
  var alu0 = (gidx0+bitcast<i32>((cast0<<10u))+bitcast<i32>((bitcast<u32>(lidx0)<<7u)));
  var val0 = data1_73728[alu0];
  var val1 = data2_576[(lidx0+bitcast<i32>((cast0<<3u)))];
  data0_73728[alu0] = (val0-val1);
}`;

const r_576_32_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_576:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_73728:array<f32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 576 */
  acc0[0] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 32; Ridx0++) {
    var alu1 = (bitcast<i32>((bitcast<u32>(gidx0)<<7u))+bitcast<i32>((bitcast<u32>(Ridx0)<<2u)));
    var val0 = data1_73728[alu1];
    var val1 = data1_73728[(alu1+1)];
    var val2 = data1_73728[(alu1+2)];
    var val3 = data1_73728[(alu1+3)];
    acc0[0] = (acc0[0]+(val0*val0)+(val1*val1)+(val2*val2)+(val3*val3));
  }
  data0_576[gidx0] = sqrt(((acc0[0]*0.0078125f)+1e-06f));
}`;

const E_8_576_16 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_73728:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_73728:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_576:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_128:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_128:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 576 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 16 */
  var alu0 = (lidx0+bitcast<i32>((bitcast<u32>(gidx1)<<4u)));
  var val0 = data1_73728[(alu0+bitcast<i32>((bitcast<u32>(gidx0)<<7u)))];
  var val1 = data2_576[gidx0];
  var val2 = data3_128[alu0];
  var val3 = data4_128[alu0];
  var alu1 = ((val0*(1/val1)*val2)+val3);
  data0_73728[(gidx0+(gidx1*9216)+(lidx0*576))] = (alu1*(1/(1.0f+exp2((alu1*-1.4426950408889634f)))));
}`;

const r_24_24_32_16_4_8_3_3 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32,64>;
@group(0) @binding(1)var<storage,read_write>data0_73728:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_73728:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_147456:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,4>;
  var acc1: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 32 */
  var gidx1 = i32(gindex.y); /* 24 */
  var gidx2 = i32(gindex.z); /* 24 */
  var lidx0 = i32(lindex.x); /* 16 */
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 8; Ridx0++) {
    for (var Ridx1 = 0; Ridx1 < 3; Ridx1++) {
      var alu4 = (gidx2+Ridx1);
      for (var Ridx2 = 0; Ridx2 < 3; Ridx2++) {
        var alu5 = (gidx1+Ridx2);
        var val0 = select(0.0f, data1_73728[(alu5+(gidx2*24)+(Ridx1*24)+(lidx0*576)+(Ridx0*9216)+-25)], ((0<alu5)&(alu5<25)&(0<alu4)&(alu4<25)));
        var alu6 = ((lidx0*9)+(Ridx0*144)+(Ridx1*3)+Ridx2+(gidx0*4608));
        var val1 = data2_147456[(alu6+1152)];
        var val2 = data2_147456[alu6];
        var val3 = data2_147456[(alu6+2304)];
        var val4 = data2_147456[(alu6+3456)];
        acc0[0] = (acc0[0]+(val0*val2));
        acc0[1] = (acc0[1]+(val0*val1));
        acc0[2] = (acc0[2]+(val0*val3));
        acc0[3] = (acc0[3]+(val0*val4));
      }
    }
  }
  var cast0 = bitcast<i32>((bitcast<u32>(lidx0)<<2u));
  temp0[cast0] = acc0[0];
  temp0[(cast0+1)] = acc0[1];
  temp0[(cast0+2)] = acc0[2];
  temp0[(cast0+3)] = acc0[3];
  workgroupBarrier();
  acc1[0] = 0.0f;
  acc1[1] = 0.0f;
  acc1[2] = 0.0f;
  acc1[3] = 0.0f;
  for (var Ridx106 = 0; Ridx106 < 16; Ridx106++) {
    var cast1 = bitcast<i32>((bitcast<u32>(Ridx106)<<2u));
    var val5 = temp0[cast1];
    var val6 = temp0[(cast1+1)];
    var val7 = temp0[(cast1+2)];
    var val8 = temp0[(cast1+3)];
    acc1[0] = (acc1[0]+val5);
    acc1[1] = (acc1[1]+val6);
    acc1[2] = (acc1[2]+val7);
    acc1[3] = (acc1[3]+val8);
  }
  var alu28 = (bitcast<i32>((bitcast<u32>(gidx0)<<2u))+bitcast<i32>((bitcast<u32>(gidx1)<<7u))+(gidx2*3072));
  var alu29 = (lidx0==0);
  if (alu29) {
    data0_73728[alu28] = acc1[0];
  }
  if (alu29) {
    data0_73728[(alu28+1)] = acc1[1];
  }
  if (alu29) {
    data0_73728[(alu28+2)] = acc1[2];
  }
  if (alu29) {
    data0_73728[(alu28+3)] = acc1[3];
  }
}`;

const E_160_36_16_2_2 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_368640:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_147456:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_73728:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_73728:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_73728:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_576:array<f32>;
@group(0) @binding(7)var<storage,read_write>data6_128:array<f32>;
@group(0) @binding(8)var<storage,read_write>data7_128:array<f32>;
@compute @workgroup_size(16,2) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 36 */
  var gidx1 = i32(gindex.y); /* 160 */
  var lidx0 = i32(lindex.x); /* 16 */
  var lidx1 = i32(lindex.y); /* 2 */
  var cast0 = bitcast<u32>(gidx0);
  var alu0 = (lidx0+bitcast<i32>((cast0<<4u)));
  var alu1 = (alu0+(gidx1*2304)+(lidx1*1152));
  var alu2 = (gidx1<32);
  var val0 = select(0.0f, data1_147456[alu1], alu2);
  var alu3 = ((31<gidx1)&(gidx1<64));
  var val1 = select(0.0f, data1_147456[alu1], alu3);
  var alu4 = ((63<gidx1)&(gidx1<96));
  var val2 = select(0.0f, data2_73728[(alu1+-147456)], alu4);
  var alu5 = (gidx1<128);
  var alu6 = ((95<gidx1)&alu5);
  var val3 = select(0.0f, data3_73728[(alu1+-221184)], alu6);
  var alu7 = (bitcast<i32>((bitcast<u32>(gidx1)<<2u))+bitcast<i32>((bitcast<u32>(lidx1)<<1u)));
  var alu8 = (bitcast<i32>((cast0<<11u))+bitcast<i32>((bitcast<u32>(lidx0)<<7u))+alu7);
  var alu9 = (127<gidx1);
  var val4 = select(0.0f, data4_73728[(alu8+-512)], alu9);
  var val5 = data5_576[alu0];
  var alu10 = (alu7+-512);
  var val6 = select(0.0f, data6_128[alu10], alu9);
  var val7 = select(0.0f, data7_128[alu10], alu9);
  var alu11 = (alu1+576);
  var val8 = select(0.0f, data1_147456[alu11], alu2);
  var val9 = select(0.0f, data1_147456[alu11], alu3);
  var val10 = select(0.0f, data2_73728[(alu1+-146880)], alu4);
  var val11 = select(0.0f, data3_73728[(alu1+-220608)], alu6);
  var val12 = select(0.0f, data4_73728[(alu8+-511)], alu9);
  var alu12 = (alu7+-511);
  var val13 = select(0.0f, data6_128[alu12], alu9);
  var val14 = select(0.0f, data7_128[alu12], alu9);
  var alu13 = (1/val5);
  var alu14 = ((val4*alu13*val6)+val7);
  var alu15 = ((val12*alu13*val13)+val14);
  var alu16 = select((alu14*(1/(1.0f+exp2((alu14*-1.4426950408889634f))))),0.0f,alu5);
  var alu17 = select((alu15*(1/(1.0f+exp2((alu15*-1.4426950408889634f))))),0.0f,alu5);
  data0_368640[alu1] = (val0+val1+val2+val3+alu16);
  data0_368640[alu11] = (val8+val9+val10+val11+alu17);
}`;

const r_6_64_32_3_4_160_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_147456:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_368640:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_163840:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,12>;
  var gidx0 = i32(gindex.x); /* 64 */
  var gidx1 = i32(gindex.y); /* 6 */
  var lidx0 = i32(lindex.x); /* 32 */
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  acc0[4] = 0.0f;
  acc0[5] = 0.0f;
  acc0[6] = 0.0f;
  acc0[7] = 0.0f;
  acc0[8] = 0.0f;
  acc0[9] = 0.0f;
  acc0[10] = 0.0f;
  acc0[11] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 160; Ridx0++) {
    var alu12 = (lidx0+(gidx1*96)+(Ridx0*2304));
    var val0 = data1_368640[alu12];
    var alu13 = ((gidx0*2560)+bitcast<i32>((bitcast<u32>(Ridx0)<<2u)));
    var val1 = data2_163840[alu13];
    var val2 = data1_368640[(alu12+576)];
    var val3 = data2_163840[(alu13+1)];
    var val4 = data1_368640[(alu12+1152)];
    var val5 = data2_163840[(alu13+2)];
    var val6 = data1_368640[(alu12+1728)];
    var val7 = data2_163840[(alu13+3)];
    var val8 = data2_163840[(alu13+640)];
    var val9 = data2_163840[(alu13+641)];
    var val10 = data2_163840[(alu13+642)];
    var val11 = data2_163840[(alu13+643)];
    var val12 = data2_163840[(alu13+1280)];
    var val13 = data2_163840[(alu13+1281)];
    var val14 = data2_163840[(alu13+1282)];
    var val15 = data2_163840[(alu13+1283)];
    var val16 = data2_163840[(alu13+1920)];
    var val17 = data2_163840[(alu13+1921)];
    var val18 = data2_163840[(alu13+1922)];
    var val19 = data2_163840[(alu13+1923)];
    var val20 = data1_368640[(alu12+32)];
    var val21 = data1_368640[(alu12+608)];
    var val22 = data1_368640[(alu12+1184)];
    var val23 = data1_368640[(alu12+1760)];
    var val24 = data1_368640[(alu12+64)];
    var val25 = data1_368640[(alu12+640)];
    var val26 = data1_368640[(alu12+1216)];
    var val27 = data1_368640[(alu12+1792)];
    acc0[0] = (acc0[0]+(val0*val1)+(val2*val3)+(val4*val5)+(val6*val7));
    acc0[1] = (acc0[1]+(val0*val8)+(val2*val9)+(val4*val10)+(val6*val11));
    acc0[2] = (acc0[2]+(val0*val12)+(val2*val13)+(val4*val14)+(val6*val15));
    acc0[3] = (acc0[3]+(val0*val16)+(val2*val17)+(val4*val18)+(val6*val19));
    acc0[4] = (acc0[4]+(val20*val1)+(val21*val3)+(val22*val5)+(val23*val7));
    acc0[5] = (acc0[5]+(val20*val8)+(val21*val9)+(val22*val10)+(val23*val11));
    acc0[6] = (acc0[6]+(val20*val12)+(val21*val13)+(val22*val14)+(val23*val15));
    acc0[7] = (acc0[7]+(val20*val16)+(val21*val17)+(val22*val18)+(val23*val19));
    acc0[8] = (acc0[8]+(val24*val1)+(val25*val3)+(val26*val5)+(val27*val7));
    acc0[9] = (acc0[9]+(val24*val8)+(val25*val9)+(val26*val10)+(val27*val11));
    acc0[10] = (acc0[10]+(val24*val12)+(val25*val13)+(val26*val14)+(val27*val15));
    acc0[11] = (acc0[11]+(val24*val16)+(val25*val17)+(val26*val18)+(val27*val19));
  }
  var alu27 = ((gidx1*24576)+bitcast<i32>((bitcast<u32>(lidx0)<<8u))+bitcast<i32>((bitcast<u32>(gidx0)<<2u)));
  data0_147456[(alu27+8192)] = acc0[4];
  data0_147456[(alu27+8193)] = acc0[5];
  data0_147456[(alu27+8194)] = acc0[6];
  data0_147456[(alu27+8195)] = acc0[7];
  data0_147456[(alu27+16384)] = acc0[8];
  data0_147456[(alu27+16385)] = acc0[9];
  data0_147456[(alu27+16386)] = acc0[10];
  data0_147456[(alu27+16387)] = acc0[11];
  data0_147456[(alu27+1)] = acc0[1];
  data0_147456[(alu27+2)] = acc0[2];
  data0_147456[(alu27+3)] = acc0[3];
  data0_147456[alu27] = acc0[0];
}`;

const E_576_16_16 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_147456:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_147456:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_576:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_256:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_256:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 576 */
  var lidx0 = i32(lindex.x); /* 16 */
  var alu0 = (lidx0+bitcast<i32>((bitcast<u32>(gidx0)<<4u)));
  var alu1 = (alu0+bitcast<i32>((bitcast<u32>(gidx1)<<8u)));
  var val0 = data1_147456[alu1];
  var val1 = data2_576[gidx1];
  var val2 = data3_256[alu0];
  var val3 = data4_256[alu0];
  var alu2 = ((val0*(1/val1)*val2)+val3);
  data0_147456[alu1] = (alu2*(1/(1.0f+exp2((alu2*-1.4426950408889634f)))));
}`;

const E_576_16_16n1 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_147456:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_147456:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_576:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_256:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_256:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 576 */
  var lidx0 = i32(lindex.x); /* 16 */
  var alu0 = (lidx0+bitcast<i32>((bitcast<u32>(gidx0)<<4u)));
  var alu1 = (alu0+bitcast<i32>((bitcast<u32>(gidx1)<<8u)));
  var val0 = data1_147456[alu1];
  var val1 = data2_576[gidx1];
  var val2 = data3_256[alu0];
  var val3 = data4_256[alu0];
  data0_147456[alu1] = ((val0*(1/val1)*val2)+val3);
}`;

const r_8_8_24_32_3_256 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_147456:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_147456:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_65536:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_256:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,3>;
  var gidx0 = i32(gindex.x); /* 24 */
  var gidx1 = i32(gindex.y); /* 8 */
  var gidx2 = i32(gindex.z); /* 8 */
  var lidx0 = i32(lindex.x); /* 32 */
  var cast0 = bitcast<u32>(gidx2);
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  for (var Ridx2 = 0; Ridx2 < 256; Ridx2++) {
    var alu3 = (bitcast<i32>((bitcast<u32>(gidx0)<<8u))+(gidx1*18432)+Ridx2);
    var val0 = data1_147456[alu3];
    var val1 = data2_65536[(bitcast<i32>((cast0<<13u))+bitcast<i32>((bitcast<u32>(lidx0)<<8u))+Ridx2)];
    var val2 = data1_147456[(alu3+6144)];
    var val3 = data1_147456[(alu3+12288)];
    acc0[0] = (acc0[0]+(val0*val1));
    acc0[1] = (acc0[1]+(val2*val1));
    acc0[2] = (acc0[2]+(val3*val1));
  }
  var val4 = data3_256[(lidx0+bitcast<i32>((cast0<<5u)))];
  var alu8 = (gidx0+(gidx1*72)+(gidx2*18432)+(lidx0*576));
  var alu9 = (gidx1*3);
  var alu10 = (16.0f*((f32((alu9+1)))+-0.5f));
  var alu11 = select((alu10+-0.5f),0.0f,(alu10<0.5f));
  var alu12 = select(alu11,383.0f,(383.0f<alu11));
  var alu13 = trunc(alu12);
  var cast1 = (i32(alu13));
  var alu14 = (16.0f*((f32((alu9+2)))+-0.5f));
  var alu15 = select((alu14+-0.5f),0.0f,(alu14<0.5f));
  var alu16 = select(alu15,383.0f,(383.0f<alu15));
  var alu17 = trunc(alu16);
  var cast2 = (i32(alu17));
  var alu18 = (16.0f*((f32((alu9+3)))+-0.5f));
  var alu19 = select((alu18+-0.5f),0.0f,(alu18<0.5f));
  var alu20 = select(alu19,383.0f,(383.0f<alu19));
  var alu21 = trunc(alu20);
  var cast3 = (i32(alu21));
  var alu22 = (alu13+-1.0f);
  var alu23 = (alu17+-1.0f);
  var alu24 = (alu21+-1.0f);
  var alu25 = (16.0f*((f32((gidx0+1)))+-0.5f));
  var alu26 = select((alu25+-0.5f),0.0f,(alu25<0.5f));
  var alu27 = select(alu26,383.0f,(383.0f<alu26));
  var alu28 = trunc(alu27);
  var alu29 = select(cast1,(i32((alu13+1.0f))),(alu13<alu12));
  var alu30 = (alu12<alu13);
  var alu31 = select(cast1,(i32(alu22)),alu30);
  var alu32 = select(alu28,(alu28+-1.0f),(alu27<alu28));
  var alu33 = (alu27-alu32);
  var alu34 = select(alu13,alu22,alu30);
  var alu35 = select(0.0f,alu33,((-1<alu29)&(alu29<384)));
  var alu36 = select(0.0f,alu33,((-1<alu31)&(alu31<384)));
  var alu37 = select(cast2,(i32((alu17+1.0f))),(alu17<alu16));
  var alu38 = (alu16<alu17);
  var alu39 = select(cast2,(i32(alu23)),alu38);
  var alu40 = select(alu17,alu23,alu38);
  var alu41 = select(0.0f,alu33,((-1<alu37)&(alu37<384)));
  var alu42 = select(0.0f,alu33,((-1<alu39)&(alu39<384)));
  var alu43 = select(cast3,(i32((alu21+1.0f))),(alu21<alu20));
  var alu44 = (alu20<alu21);
  var alu45 = select(cast3,(i32(alu24)),alu44);
  var alu46 = select(alu21,alu24,alu44);
  var alu47 = select(0.0f,alu33,((-1<alu43)&(alu43<384)));
  var alu48 = select(0.0f,alu33,((-1<alu45)&(alu45<384)));
  var alu49 = select(0.0f,(acc0[0]+val4),((alu36+((alu35-alu36)*(alu12-alu34)))!=0.0f));
  var alu50 = select(0.0f,(acc0[1]+val4),((alu42+((alu41-alu42)*(alu16-alu40)))!=0.0f));
  var alu51 = select(0.0f,(acc0[2]+val4),((alu48+((alu47-alu48)*(alu20-alu46)))!=0.0f));
  data0_147456[alu8] = alu49;
  data0_147456[(alu8+24)] = alu50;
  data0_147456[(alu8+48)] = alu51;
}`;

const r_24_2_16_2_8_12_24_24_4_64_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_147456:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_147456:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_65536:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_256:array<f32>;
@compute @workgroup_size(16,2,8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<i32,1>;
  var acc1: array<i32,1>;
  var acc2: array<bool,12>;
  var acc3: array<f32,12>;
  acc0[0] = 0;
  for (var Ridx2 = 0; Ridx2 < 24; Ridx2++) {
    var alu1 = (16.0f*((f32((Ridx2+1)))+-0.5f));
    var alu2 = select((alu1+-0.5f),0.0f,(alu1<0.5f));
    var alu3 = select(alu2,383.0f,(383.0f<alu2));
    var alu4 = trunc(alu3);
    var alu5 = select(alu4,(alu4+-1.0f),(alu3<alu4));
    acc0[0] = (acc0[0]+(i32(((alu3-alu5)!=0.0f))));
  }
  acc1[0] = 0;
  for (var Ridx3 = 0; Ridx3 < 24; Ridx3++) {
    var alu9 = (16.0f*((f32((Ridx3+1)))+-0.5f));
    var alu10 = select((alu9+-0.5f),0.0f,(alu9<0.5f));
    var alu11 = select(alu10,383.0f,(383.0f<alu10));
    var alu12 = trunc(alu11);
    var cast0 = (i32(alu12));
    var alu13 = (alu12+-1.0f);
    var alu14 = select(cast0,(i32((alu12+1.0f))),(alu12<alu11));
    var alu15 = (alu11<alu12);
    var alu16 = select(cast0,(i32(alu13)),alu15);
    var alu17 = select(alu12,alu13,alu15);
    var alu18 = select(0.0f,0.5f,((-1<alu14)&(alu14<384)));
    var alu19 = select(0.0f,0.5f,((-1<alu16)&(alu16<384)));
    acc1[0] = (acc1[0]+(i32(((alu19+((alu18-alu19)*(alu11-alu17)))!=0.0f))));
  }
  var gidx1 = i32(gindex.y); /* 24 */
  var lidx1 = i32(lindex.y); /* 2 */
  acc2[0] = false;
  acc2[1] = false;
  acc2[2] = false;
  acc2[3] = false;
  acc2[4] = false;
  acc2[5] = false;
  acc2[6] = false;
  acc2[7] = false;
  acc2[8] = false;
  acc2[9] = false;
  acc2[10] = false;
  acc2[11] = false;
  for (var Ridx4 = 0; Ridx4 < 4; Ridx4++) {
    var alu34 = (Ridx4<1);
    var alu35 = select(0,acc0[0],alu34);
    var alu36 = select(acc1[0],0,alu34);
    var alu37 = (1/(f32((alu35+alu36))));
    var alu38 = (Ridx4==1);
    var alu39 = select(0.0f,(f32(gidx1)),alu34);
    var alu40 = select(0.0f,(f32(lidx1)),alu38);
    var alu41 = select(0.0f,(f32((lidx1+2))),alu38);
    var alu42 = select(0.0f,(f32((lidx1+4))),alu38);
    var alu43 = select(0.0f,(f32((lidx1+6))),alu38);
    var alu44 = select(0.0f,(f32((lidx1+8))),alu38);
    var alu45 = select(0.0f,(f32((lidx1+10))),alu38);
    var alu46 = select(0.0f,(f32((lidx1+12))),alu38);
    var alu47 = select(0.0f,(f32((lidx1+14))),alu38);
    var alu48 = select(0.0f,(f32((lidx1+16))),alu38);
    var alu49 = select(0.0f,(f32((lidx1+18))),alu38);
    var alu50 = select(0.0f,(f32((lidx1+20))),alu38);
    var alu51 = select(0.0f,(f32((lidx1+22))),alu38);
    var alu52 = (Ridx4<2);
    var alu53 = select(0.05f,((alu39+alu40+0.5f)*alu37),alu52);
    var alu54 = select(0.05f,((alu39+alu41+0.5f)*alu37),alu52);
    var alu55 = select(0.05f,((alu39+alu42+0.5f)*alu37),alu52);
    var alu56 = select(0.05f,((alu39+alu43+0.5f)*alu37),alu52);
    var alu57 = select(0.05f,((alu39+alu44+0.5f)*alu37),alu52);
    var alu58 = select(0.05f,((alu39+alu45+0.5f)*alu37),alu52);
    var alu59 = select(0.05f,((alu39+alu46+0.5f)*alu37),alu52);
    var alu60 = select(0.05f,((alu39+alu47+0.5f)*alu37),alu52);
    var alu61 = select(0.05f,((alu39+alu48+0.5f)*alu37),alu52);
    var alu62 = select(0.05f,((alu39+alu49+0.5f)*alu37),alu52);
    var alu63 = select(0.05f,((alu39+alu50+0.5f)*alu37),alu52);
    var alu64 = select(0.05f,((alu39+alu51+0.5f)*alu37),alu52);
    acc2[0] = (acc2[0]|(((0.01f<alu53)&(alu53<0.99f))!=true));
    acc2[1] = (acc2[1]|(((0.01f<alu54)&(alu54<0.99f))!=true));
    acc2[2] = (acc2[2]|(((0.01f<alu55)&(alu55<0.99f))!=true));
    acc2[3] = (acc2[3]|(((0.01f<alu56)&(alu56<0.99f))!=true));
    acc2[4] = (acc2[4]|(((0.01f<alu57)&(alu57<0.99f))!=true));
    acc2[5] = (acc2[5]|(((0.01f<alu58)&(alu58<0.99f))!=true));
    acc2[6] = (acc2[6]|(((0.01f<alu59)&(alu59<0.99f))!=true));
    acc2[7] = (acc2[7]|(((0.01f<alu60)&(alu60<0.99f))!=true));
    acc2[8] = (acc2[8]|(((0.01f<alu61)&(alu61<0.99f))!=true));
    acc2[9] = (acc2[9]|(((0.01f<alu62)&(alu62<0.99f))!=true));
    acc2[10] = (acc2[10]|(((0.01f<alu63)&(alu63<0.99f))!=true));
    acc2[11] = (acc2[11]|(((0.01f<alu64)&(alu64<0.99f))!=true));
  }
  var gidx0 = i32(gindex.x); /* 2 */
  var lidx0 = i32(lindex.x); /* 16 */
  var lidx2 = i32(lindex.z); /* 8 */
  var alu78 = (16.0f*((f32((lidx1+1)))+-0.5f));
  var alu79 = select((alu78+-0.5f),0.0f,(alu78<0.5f));
  var alu80 = select(alu79,383.0f,(383.0f<alu79));
  var alu81 = trunc(alu80);
  var cast1 = (i32(alu81));
  var alu82 = (16.0f*((f32((lidx1+3)))+-0.5f));
  var alu83 = select((alu82+-0.5f),0.0f,(alu82<0.5f));
  var alu84 = select(alu83,383.0f,(383.0f<alu83));
  var alu85 = trunc(alu84);
  var cast2 = (i32(alu85));
  var alu86 = (16.0f*((f32((lidx1+5)))+-0.5f));
  var alu87 = select((alu86+-0.5f),0.0f,(alu86<0.5f));
  var alu88 = select(alu87,383.0f,(383.0f<alu87));
  var alu89 = trunc(alu88);
  var cast3 = (i32(alu89));
  var alu90 = (16.0f*((f32((lidx1+7)))+-0.5f));
  var alu91 = select((alu90+-0.5f),0.0f,(alu90<0.5f));
  var alu92 = select(alu91,383.0f,(383.0f<alu91));
  var alu93 = trunc(alu92);
  var cast4 = (i32(alu93));
  var alu94 = (16.0f*((f32((lidx1+9)))+-0.5f));
  var alu95 = select((alu94+-0.5f),0.0f,(alu94<0.5f));
  var alu96 = select(alu95,383.0f,(383.0f<alu95));
  var alu97 = trunc(alu96);
  var cast5 = (i32(alu97));
  var alu98 = (16.0f*((f32((lidx1+11)))+-0.5f));
  var alu99 = select((alu98+-0.5f),0.0f,(alu98<0.5f));
  var alu100 = select(alu99,383.0f,(383.0f<alu99));
  var alu101 = trunc(alu100);
  var cast6 = (i32(alu101));
  var alu102 = (16.0f*((f32((lidx1+13)))+-0.5f));
  var alu103 = select((alu102+-0.5f),0.0f,(alu102<0.5f));
  var alu104 = select(alu103,383.0f,(383.0f<alu103));
  var alu105 = trunc(alu104);
  var cast7 = (i32(alu105));
  var alu106 = (16.0f*((f32((lidx1+15)))+-0.5f));
  var alu107 = select((alu106+-0.5f),0.0f,(alu106<0.5f));
  var alu108 = select(alu107,383.0f,(383.0f<alu107));
  var alu109 = trunc(alu108);
  var cast8 = (i32(alu109));
  var alu110 = (16.0f*((f32((lidx1+17)))+-0.5f));
  var alu111 = select((alu110+-0.5f),0.0f,(alu110<0.5f));
  var alu112 = select(alu111,383.0f,(383.0f<alu111));
  var alu113 = trunc(alu112);
  var cast9 = (i32(alu113));
  var alu114 = (16.0f*((f32((lidx1+19)))+-0.5f));
  var alu115 = select((alu114+-0.5f),0.0f,(alu114<0.5f));
  var alu116 = select(alu115,383.0f,(383.0f<alu115));
  var alu117 = trunc(alu116);
  var cast10 = (i32(alu117));
  var alu118 = (16.0f*((f32((lidx1+21)))+-0.5f));
  var alu119 = select((alu118+-0.5f),0.0f,(alu118<0.5f));
  var alu120 = select(alu119,383.0f,(383.0f<alu119));
  var alu121 = trunc(alu120);
  var cast11 = (i32(alu121));
  var alu122 = (16.0f*((f32((lidx1+23)))+-0.5f));
  var alu123 = select((alu122+-0.5f),0.0f,(alu122<0.5f));
  var alu124 = select(alu123,383.0f,(383.0f<alu123));
  var alu125 = trunc(alu124);
  var cast12 = (i32(alu125));
  var alu126 = (alu81+-1.0f);
  var alu127 = (alu85+-1.0f);
  var alu128 = (alu89+-1.0f);
  var alu129 = (alu93+-1.0f);
  var alu130 = (alu97+-1.0f);
  var alu131 = (alu101+-1.0f);
  var alu132 = (alu105+-1.0f);
  var alu133 = (alu109+-1.0f);
  var alu134 = (alu113+-1.0f);
  var alu135 = (alu117+-1.0f);
  var alu136 = (alu121+-1.0f);
  var alu137 = (alu125+-1.0f);
  var cast13 = bitcast<u32>(gidx0);
  var cast14 = bitcast<u32>(lidx2);
  var alu138 = (16.0f*((f32((gidx1+1)))+-0.5f));
  var alu139 = select((alu138+-0.5f),0.0f,(alu138<0.5f));
  var alu140 = select(alu139,383.0f,(383.0f<alu139));
  var alu141 = trunc(alu140);
  var alu142 = (bitcast<i32>((bitcast<u32>(gidx1)<<8u))+(lidx1*6144));
  var alu143 = select(cast1,(i32((alu81+1.0f))),(alu81<alu80));
  var alu144 = (alu80<alu81);
  var alu145 = select(cast1,(i32(alu126)),alu144);
  var alu146 = select(alu141,(alu141+-1.0f),(alu140<alu141));
  var alu147 = (alu140-alu146);
  var alu148 = select(alu81,alu126,alu144);
  var alu149 = select(0.0f,alu147,((-1<alu143)&(alu143<384)));
  var alu150 = select(0.0f,alu147,((-1<alu145)&(alu145<384)));
  var alu151 = select(cast2,(i32((alu85+1.0f))),(alu85<alu84));
  var alu152 = (alu84<alu85);
  var alu153 = select(cast2,(i32(alu127)),alu152);
  var alu154 = select(alu85,alu127,alu152);
  var alu155 = select(0.0f,alu147,((-1<alu151)&(alu151<384)));
  var alu156 = select(0.0f,alu147,((-1<alu153)&(alu153<384)));
  var alu157 = select(cast3,(i32((alu89+1.0f))),(alu89<alu88));
  var alu158 = (alu88<alu89);
  var alu159 = select(cast3,(i32(alu128)),alu158);
  var alu160 = select(alu89,alu128,alu158);
  var alu161 = select(0.0f,alu147,((-1<alu157)&(alu157<384)));
  var alu162 = select(0.0f,alu147,((-1<alu159)&(alu159<384)));
  var alu163 = select(cast4,(i32((alu93+1.0f))),(alu93<alu92));
  var alu164 = (alu92<alu93);
  var alu165 = select(cast4,(i32(alu129)),alu164);
  var alu166 = select(alu93,alu129,alu164);
  var alu167 = select(0.0f,alu147,((-1<alu163)&(alu163<384)));
  var alu168 = select(0.0f,alu147,((-1<alu165)&(alu165<384)));
  var alu169 = select(cast5,(i32((alu97+1.0f))),(alu97<alu96));
  var alu170 = (alu96<alu97);
  var alu171 = select(cast5,(i32(alu130)),alu170);
  var alu172 = select(alu97,alu130,alu170);
  var alu173 = select(0.0f,alu147,((-1<alu169)&(alu169<384)));
  var alu174 = select(0.0f,alu147,((-1<alu171)&(alu171<384)));
  var alu175 = select(cast6,(i32((alu101+1.0f))),(alu101<alu100));
  var alu176 = (alu100<alu101);
  var alu177 = select(cast6,(i32(alu131)),alu176);
  var alu178 = select(alu101,alu131,alu176);
  var alu179 = select(0.0f,alu147,((-1<alu175)&(alu175<384)));
  var alu180 = select(0.0f,alu147,((-1<alu177)&(alu177<384)));
  var alu181 = select(cast7,(i32((alu105+1.0f))),(alu105<alu104));
  var alu182 = (alu104<alu105);
  var alu183 = select(cast7,(i32(alu132)),alu182);
  var alu184 = select(alu105,alu132,alu182);
  var alu185 = select(0.0f,alu147,((-1<alu181)&(alu181<384)));
  var alu186 = select(0.0f,alu147,((-1<alu183)&(alu183<384)));
  var alu187 = select(cast8,(i32((alu109+1.0f))),(alu109<alu108));
  var alu188 = (alu108<alu109);
  var alu189 = select(cast8,(i32(alu133)),alu188);
  var alu190 = select(alu109,alu133,alu188);
  var alu191 = select(0.0f,alu147,((-1<alu187)&(alu187<384)));
  var alu192 = select(0.0f,alu147,((-1<alu189)&(alu189<384)));
  var alu193 = select(cast9,(i32((alu113+1.0f))),(alu113<alu112));
  var alu194 = (alu112<alu113);
  var alu195 = select(cast9,(i32(alu134)),alu194);
  var alu196 = select(alu113,alu134,alu194);
  var alu197 = select(0.0f,alu147,((-1<alu193)&(alu193<384)));
  var alu198 = select(0.0f,alu147,((-1<alu195)&(alu195<384)));
  var alu199 = select(cast10,(i32((alu117+1.0f))),(alu117<alu116));
  var alu200 = (alu116<alu117);
  var alu201 = select(cast10,(i32(alu135)),alu200);
  var alu202 = select(alu117,alu135,alu200);
  var alu203 = select(0.0f,alu147,((-1<alu199)&(alu199<384)));
  var alu204 = select(0.0f,alu147,((-1<alu201)&(alu201<384)));
  var alu205 = select(cast11,(i32((alu121+1.0f))),(alu121<alu120));
  var alu206 = (alu120<alu121);
  var alu207 = select(cast11,(i32(alu136)),alu206);
  var alu208 = select(alu121,alu136,alu206);
  var alu209 = select(0.0f,alu147,((-1<alu205)&(alu205<384)));
  var alu210 = select(0.0f,alu147,((-1<alu207)&(alu207<384)));
  var alu211 = select(cast12,(i32((alu125+1.0f))),(alu125<alu124));
  var alu212 = (alu124<alu125);
  var alu213 = select(cast12,(i32(alu137)),alu212);
  var alu214 = select(alu125,alu137,alu212);
  var alu215 = select(0.0f,alu147,((-1<alu211)&(alu211<384)));
  var alu216 = select(0.0f,alu147,((-1<alu213)&(alu213<384)));
  var alu217 = (((alu150+((alu149-alu150)*(alu80-alu148)))!=0.0f)&(acc2[0]!=true));
  var alu218 = (((alu156+((alu155-alu156)*(alu84-alu154)))!=0.0f)&(acc2[1]!=true));
  var alu219 = (((alu162+((alu161-alu162)*(alu88-alu160)))!=0.0f)&(acc2[2]!=true));
  var alu220 = (((alu168+((alu167-alu168)*(alu92-alu166)))!=0.0f)&(acc2[3]!=true));
  var alu221 = (((alu174+((alu173-alu174)*(alu96-alu172)))!=0.0f)&(acc2[4]!=true));
  var alu222 = (((alu180+((alu179-alu180)*(alu100-alu178)))!=0.0f)&(acc2[5]!=true));
  var alu223 = (((alu186+((alu185-alu186)*(alu104-alu184)))!=0.0f)&(acc2[6]!=true));
  var alu224 = (((alu192+((alu191-alu192)*(alu108-alu190)))!=0.0f)&(acc2[7]!=true));
  var alu225 = (((alu198+((alu197-alu198)*(alu112-alu196)))!=0.0f)&(acc2[8]!=true));
  var alu226 = (((alu204+((alu203-alu204)*(alu116-alu202)))!=0.0f)&(acc2[9]!=true));
  var alu227 = (((alu210+((alu209-alu210)*(alu120-alu208)))!=0.0f)&(acc2[10]!=true));
  var alu228 = (((alu216+((alu215-alu216)*(alu124-alu214)))!=0.0f)&(acc2[11]!=true));
  acc3[0] = 0.0f;
  acc3[1] = 0.0f;
  acc3[2] = 0.0f;
  acc3[3] = 0.0f;
  acc3[4] = 0.0f;
  acc3[5] = 0.0f;
  acc3[6] = 0.0f;
  acc3[7] = 0.0f;
  acc3[8] = 0.0f;
  acc3[9] = 0.0f;
  acc3[10] = 0.0f;
  acc3[11] = 0.0f;
  for (var Ridx5 = 0; Ridx5 < 64; Ridx5++) {
    var cast15 = bitcast<i32>((bitcast<u32>(Ridx5)<<2u));
    var alu241 = (alu142+cast15);
    var val0 = select(0.0f, data1_147456[alu241], alu217);
    var alu242 = (bitcast<i32>((cast13<<15u))+bitcast<i32>((cast14<<12u))+bitcast<i32>((bitcast<u32>(lidx0)<<8u))+cast15);
    var val1 = data2_65536[alu242];
    var val2 = select(0.0f, data1_147456[(alu241+1)], alu217);
    var val3 = data2_65536[(alu242+1)];
    var val4 = select(0.0f, data1_147456[(alu241+2)], alu217);
    var val5 = data2_65536[(alu242+2)];
    var val6 = select(0.0f, data1_147456[(alu241+3)], alu217);
    var val7 = data2_65536[(alu242+3)];
    var val8 = select(0.0f, data1_147456[(alu241+12288)], alu218);
    var val9 = select(0.0f, data1_147456[(alu241+12289)], alu218);
    var val10 = select(0.0f, data1_147456[(alu241+12290)], alu218);
    var val11 = select(0.0f, data1_147456[(alu241+12291)], alu218);
    var val12 = select(0.0f, data1_147456[(alu241+24576)], alu219);
    var val13 = select(0.0f, data1_147456[(alu241+24577)], alu219);
    var val14 = select(0.0f, data1_147456[(alu241+24578)], alu219);
    var val15 = select(0.0f, data1_147456[(alu241+24579)], alu219);
    var val16 = select(0.0f, data1_147456[(alu241+36864)], alu220);
    var val17 = select(0.0f, data1_147456[(alu241+36865)], alu220);
    var val18 = select(0.0f, data1_147456[(alu241+36866)], alu220);
    var val19 = select(0.0f, data1_147456[(alu241+36867)], alu220);
    var val20 = select(0.0f, data1_147456[(alu241+49152)], alu221);
    var val21 = select(0.0f, data1_147456[(alu241+49153)], alu221);
    var val22 = select(0.0f, data1_147456[(alu241+49154)], alu221);
    var val23 = select(0.0f, data1_147456[(alu241+49155)], alu221);
    var val24 = select(0.0f, data1_147456[(alu241+61440)], alu222);
    var val25 = select(0.0f, data1_147456[(alu241+61441)], alu222);
    var val26 = select(0.0f, data1_147456[(alu241+61442)], alu222);
    var val27 = select(0.0f, data1_147456[(alu241+61443)], alu222);
    var val28 = select(0.0f, data1_147456[(alu241+73728)], alu223);
    var val29 = select(0.0f, data1_147456[(alu241+73729)], alu223);
    var val30 = select(0.0f, data1_147456[(alu241+73730)], alu223);
    var val31 = select(0.0f, data1_147456[(alu241+73731)], alu223);
    var val32 = select(0.0f, data1_147456[(alu241+86016)], alu224);
    var val33 = select(0.0f, data1_147456[(alu241+86017)], alu224);
    var val34 = select(0.0f, data1_147456[(alu241+86018)], alu224);
    var val35 = select(0.0f, data1_147456[(alu241+86019)], alu224);
    var val36 = select(0.0f, data1_147456[(alu241+98304)], alu225);
    var val37 = select(0.0f, data1_147456[(alu241+98305)], alu225);
    var val38 = select(0.0f, data1_147456[(alu241+98306)], alu225);
    var val39 = select(0.0f, data1_147456[(alu241+98307)], alu225);
    var val40 = select(0.0f, data1_147456[(alu241+110592)], alu226);
    var val41 = select(0.0f, data1_147456[(alu241+110593)], alu226);
    var val42 = select(0.0f, data1_147456[(alu241+110594)], alu226);
    var val43 = select(0.0f, data1_147456[(alu241+110595)], alu226);
    var val44 = select(0.0f, data1_147456[(alu241+122880)], alu227);
    var val45 = select(0.0f, data1_147456[(alu241+122881)], alu227);
    var val46 = select(0.0f, data1_147456[(alu241+122882)], alu227);
    var val47 = select(0.0f, data1_147456[(alu241+122883)], alu227);
    var val48 = select(0.0f, data1_147456[(alu241+135168)], alu228);
    var val49 = select(0.0f, data1_147456[(alu241+135169)], alu228);
    var val50 = select(0.0f, data1_147456[(alu241+135170)], alu228);
    var val51 = select(0.0f, data1_147456[(alu241+135171)], alu228);
    acc3[0] = (acc3[0]+(val0*val1)+(val2*val3)+(val4*val5)+(val6*val7));
    acc3[1] = (acc3[1]+(val8*val1)+(val9*val3)+(val10*val5)+(val11*val7));
    acc3[2] = (acc3[2]+(val12*val1)+(val13*val3)+(val14*val5)+(val15*val7));
    acc3[3] = (acc3[3]+(val16*val1)+(val17*val3)+(val18*val5)+(val19*val7));
    acc3[4] = (acc3[4]+(val20*val1)+(val21*val3)+(val22*val5)+(val23*val7));
    acc3[5] = (acc3[5]+(val24*val1)+(val25*val3)+(val26*val5)+(val27*val7));
    acc3[6] = (acc3[6]+(val28*val1)+(val29*val3)+(val30*val5)+(val31*val7));
    acc3[7] = (acc3[7]+(val32*val1)+(val33*val3)+(val34*val5)+(val35*val7));
    acc3[8] = (acc3[8]+(val36*val1)+(val37*val3)+(val38*val5)+(val39*val7));
    acc3[9] = (acc3[9]+(val40*val1)+(val41*val3)+(val42*val5)+(val43*val7));
    acc3[10] = (acc3[10]+(val44*val1)+(val45*val3)+(val46*val5)+(val47*val7));
    acc3[11] = (acc3[11]+(val48*val1)+(val49*val3)+(val50*val5)+(val51*val7));
  }
  var alu256 = (lidx0+bitcast<i32>((cast13<<7u))+bitcast<i32>((cast14<<4u)));
  var val52 = data3_256[alu256];
  var alu257 = (alu256+alu142);
  data0_147456[alu257] = (acc3[0]+val52);
  data0_147456[(alu257+12288)] = (acc3[1]+val52);
  data0_147456[(alu257+24576)] = (acc3[2]+val52);
  data0_147456[(alu257+36864)] = (acc3[3]+val52);
  data0_147456[(alu257+49152)] = (acc3[4]+val52);
  data0_147456[(alu257+61440)] = (acc3[5]+val52);
  data0_147456[(alu257+73728)] = (acc3[6]+val52);
  data0_147456[(alu257+86016)] = (acc3[7]+val52);
  data0_147456[(alu257+98304)] = (acc3[8]+val52);
  data0_147456[(alu257+110592)] = (acc3[9]+val52);
  data0_147456[(alu257+122880)] = (acc3[10]+val52);
  data0_147456[(alu257+135168)] = (acc3[11]+val52);
}`;

const r_576_16_16n2 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32,16>;
@group(0) @binding(1)var<storage,read_write>data0_576:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_147456:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,1>;
  var acc1: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 576 */
  var lidx0 = i32(lindex.x); /* 16 */
  acc0[0] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 16; Ridx0++) {
    var val0 = data1_147456[(bitcast<i32>((bitcast<u32>(lidx0)<<4u))+Ridx0+bitcast<i32>((bitcast<u32>(gidx0)<<8u)))];
    acc0[0] = (acc0[0]+val0);
  }
  temp0[lidx0] = acc0[0];
  workgroupBarrier();
  acc1[0] = 0.0f;
  for (var Ridx102 = 0; Ridx102 < 16; Ridx102++) {
    var val1 = temp0[Ridx102];
    acc1[0] = (acc1[0]+val1);
  }
  var alu8 = (lidx0==0);
  if (alu8) {
    data0_576[gidx0] = (acc1[0]*0.00390625f);
  }
}`;

const r_576_16_16n3 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32,16>;
@group(0) @binding(1)var<storage,read_write>data0_576:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_147456:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_576:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,1>;
  var acc1: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 576 */
  var val0 = data2_576[gidx0];
  var lidx0 = i32(lindex.x); /* 16 */
  acc0[0] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 16; Ridx0++) {
    var val1 = data1_147456[(bitcast<i32>((bitcast<u32>(lidx0)<<4u))+Ridx0+bitcast<i32>((bitcast<u32>(gidx0)<<8u)))];
    var alu1 = (val1-val0);
    acc0[0] = (acc0[0]+(alu1*alu1));
  }
  temp0[lidx0] = acc0[0];
  workgroupBarrier();
  acc1[0] = 0.0f;
  for (var Ridx102 = 0; Ridx102 < 16; Ridx102++) {
    var val2 = temp0[Ridx102];
    acc1[0] = (acc1[0]+val2);
  }
  var alu9 = (lidx0==0);
  if (alu9) {
    data0_576[gidx0] = (1/sqrt(((acc1[0]*0.00390625f)+1e-05f)));
  }
}`;

const E_72_64_8_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_147456:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_147456:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_576:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_576:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_256:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_256:array<f32>;
@compute @workgroup_size(8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 64 */
  var gidx1 = i32(gindex.y); /* 72 */
  var lidx0 = i32(lindex.x); /* 8 */
  var cast0 = bitcast<i32>((bitcast<u32>(gidx0)<<2u));
  var cast1 = bitcast<u32>(gidx1);
  var alu0 = (bitcast<i32>((cast1<<11u))+bitcast<i32>((bitcast<u32>(lidx0)<<8u))+cast0);
  var val0 = data1_147456[alu0];
  var alu1 = (lidx0+bitcast<i32>((cast1<<3u)));
  var val1 = data2_576[alu1];
  var val2 = data3_576[alu1];
  var val3 = data4_256[cast0];
  var val4 = data5_256[cast0];
  var alu2 = (alu0+1);
  var val5 = data1_147456[alu2];
  var alu3 = (cast0+1);
  var val6 = data4_256[alu3];
  var alu4 = (cast0+2);
  var val7 = data4_256[alu4];
  var val8 = data5_256[alu3];
  var alu5 = (alu0+2);
  var val9 = data1_147456[alu5];
  var val10 = data5_256[alu4];
  var alu6 = (alu0+3);
  var val11 = data1_147456[alu6];
  var alu7 = (cast0+3);
  var val12 = data4_256[alu7];
  var val13 = data5_256[alu7];
  data0_147456[alu0] = (((val0-val1)*val2*val3)+val4);
  data0_147456[alu2] = (((val5-val1)*val2*val6)+val8);
  data0_147456[alu5] = (((val9-val1)*val2*val7)+val10);
  data0_147456[alu6] = (((val11-val1)*val2*val12)+val13);
}`;

const r_48_91_3_4_256 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32,1092>;
@group(0) @binding(1)var<storage,read_write>data0_576:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_147456:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_23296:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_91:array<f32>;
@compute @workgroup_size(91) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,12>;
  var acc1: array<f32,12>;
  var gidx0 = i32(gindex.x); /* 48 */
  var lidx0 = i32(lindex.x); /* 91 */
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  acc0[4] = 0.0f;
  acc0[5] = 0.0f;
  acc0[6] = 0.0f;
  acc0[7] = 0.0f;
  acc0[8] = 0.0f;
  acc0[9] = 0.0f;
  acc0[10] = 0.0f;
  acc0[11] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 256; Ridx0++) {
    var alu12 = ((gidx0*3072)+Ridx0);
    var val0 = data1_147456[alu12];
    var val1 = data2_23296[(bitcast<i32>((bitcast<u32>(lidx0)<<8u))+Ridx0)];
    var val2 = data1_147456[(alu12+768)];
    var val3 = data1_147456[(alu12+1536)];
    var val4 = data1_147456[(alu12+256)];
    var val5 = data1_147456[(alu12+2304)];
    var val6 = data1_147456[(alu12+1024)];
    var val7 = data1_147456[(alu12+1792)];
    var val8 = data1_147456[(alu12+2560)];
    var val9 = data1_147456[(alu12+512)];
    var val10 = data1_147456[(alu12+1280)];
    var val11 = data1_147456[(alu12+2048)];
    var val12 = data1_147456[(alu12+2816)];
    acc0[0] = (acc0[0]+(val0*val1));
    acc0[1] = (acc0[1]+(val2*val1));
    acc0[2] = (acc0[2]+(val3*val1));
    acc0[3] = (acc0[3]+(val5*val1));
    acc0[4] = (acc0[4]+(val4*val1));
    acc0[5] = (acc0[5]+(val6*val1));
    acc0[6] = (acc0[6]+(val7*val1));
    acc0[7] = (acc0[7]+(val8*val1));
    acc0[8] = (acc0[8]+(val9*val1));
    acc0[9] = (acc0[9]+(val10*val1));
    acc0[10] = (acc0[10]+(val11*val1));
    acc0[11] = (acc0[11]+(val12*val1));
  }
  var val13 = data3_91[lidx0];
  var alu26 = (lidx0*12);
  temp0[(alu26+1)] = (acc0[1]+val13);
  temp0[(alu26+2)] = (acc0[2]+val13);
  temp0[(alu26+3)] = (acc0[3]+val13);
  temp0[(alu26+4)] = (acc0[4]+val13);
  temp0[(alu26+5)] = (acc0[5]+val13);
  temp0[(alu26+6)] = (acc0[6]+val13);
  temp0[(alu26+7)] = (acc0[7]+val13);
  temp0[(alu26+8)] = (acc0[8]+val13);
  temp0[(alu26+9)] = (acc0[9]+val13);
  temp0[(alu26+10)] = (acc0[10]+val13);
  temp0[(alu26+11)] = (acc0[11]+val13);
  temp0[alu26] = (acc0[0]+val13);
  workgroupBarrier();
  acc1[0] = (f32(-INFINITY));
  acc1[1] = (f32(-INFINITY));
  acc1[2] = (f32(-INFINITY));
  acc1[3] = (f32(-INFINITY));
  acc1[4] = (f32(-INFINITY));
  acc1[5] = (f32(-INFINITY));
  acc1[6] = (f32(-INFINITY));
  acc1[7] = (f32(-INFINITY));
  acc1[8] = (f32(-INFINITY));
  acc1[9] = (f32(-INFINITY));
  acc1[10] = (f32(-INFINITY));
  acc1[11] = (f32(-INFINITY));
  for (var Ridx103 = 0; Ridx103 < 91; Ridx103++) {
    var alu52 = (Ridx103*12);
    var val14 = temp0[alu52];
    var val15 = temp0[(alu52+1)];
    var val16 = temp0[(alu52+2)];
    var val17 = temp0[(alu52+3)];
    var val18 = temp0[(alu52+4)];
    var val19 = temp0[(alu52+5)];
    var val20 = temp0[(alu52+6)];
    var val21 = temp0[(alu52+7)];
    var val22 = temp0[(alu52+8)];
    var val23 = temp0[(alu52+9)];
    var val24 = temp0[(alu52+10)];
    var val25 = temp0[(alu52+11)];
    var alu53 = select(acc1[0],val14,(acc1[0]<val14));
    var alu54 = select(acc1[1],val15,(acc1[1]<val15));
    var alu55 = select(acc1[2],val16,(acc1[2]<val16));
    var alu56 = select(acc1[3],val17,(acc1[3]<val17));
    var alu57 = select(acc1[4],val18,(acc1[4]<val18));
    var alu58 = select(acc1[5],val19,(acc1[5]<val19));
    var alu59 = select(acc1[6],val20,(acc1[6]<val20));
    var alu60 = select(acc1[7],val21,(acc1[7]<val21));
    var alu61 = select(acc1[8],val22,(acc1[8]<val22));
    var alu62 = select(acc1[9],val23,(acc1[9]<val23));
    var alu63 = select(acc1[10],val24,(acc1[10]<val24));
    var alu64 = select(acc1[11],val25,(acc1[11]<val25));
    acc1[0] = alu53;
    acc1[1] = alu54;
    acc1[2] = alu55;
    acc1[3] = alu56;
    acc1[4] = alu57;
    acc1[5] = alu58;
    acc1[6] = alu59;
    acc1[7] = alu60;
    acc1[8] = alu61;
    acc1[9] = alu62;
    acc1[10] = alu63;
    acc1[11] = alu64;
  }
  var alu78 = (gidx0*12);
  var alu79 = (lidx0==0);
  if (alu79) {
    data0_576[(alu78+1)] = acc1[4];
  }
  if (alu79) {
    data0_576[(alu78+2)] = acc1[8];
  }
  if (alu79) {
    data0_576[(alu78+3)] = acc1[1];
  }
  if (alu79) {
    data0_576[(alu78+4)] = acc1[5];
  }
  if (alu79) {
    data0_576[(alu78+5)] = acc1[9];
  }
  if (alu79) {
    data0_576[(alu78+6)] = acc1[2];
  }
  if (alu79) {
    data0_576[(alu78+7)] = acc1[6];
  }
  if (alu79) {
    data0_576[(alu78+8)] = acc1[10];
  }
  if (alu79) {
    data0_576[(alu78+9)] = acc1[3];
  }
  if (alu79) {
    data0_576[(alu78+10)] = acc1[7];
  }
  if (alu79) {
    data0_576[(alu78+11)] = acc1[11];
  }
  if (alu79) {
    data0_576[alu78] = acc1[0];
  }
}`;

const r_144_16_16_4_256 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_147456:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_147456:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_65536:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_256:array<f32>;
@compute @workgroup_size(16,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 144 */
  var lidx0 = i32(lindex.x); /* 16 */
  var lidx1 = i32(lindex.y); /* 4 */
  var cast0 = bitcast<u32>(gidx0);
  var alu0 = (bitcast<i32>((bitcast<u32>(gidx1)<<10u))+bitcast<i32>((bitcast<u32>(lidx1)<<8u)));
  acc0[0] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 256; Ridx0++) {
    var val0 = data1_147456[(alu0+Ridx0)];
    var val1 = data2_65536[(bitcast<i32>((cast0<<12u))+bitcast<i32>((bitcast<u32>(lidx0)<<8u))+Ridx0)];
    acc0[0] = (acc0[0]+(val0*val1));
  }
  var alu4 = (lidx0+bitcast<i32>((cast0<<4u)));
  var val2 = data3_256[alu4];
  var alu5 = (acc0[0]+val2);
  var alu6 = select(0.0f,alu5,(0.0f<alu5));
  data0_147456[(alu4+alu0)] = alu6;
}`;

const E_256_2_2 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_1024:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_576:array<f32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 2 */
  var gidx1 = i32(gindex.y); /* 256 */
  var cast0 = bitcast<i32>((bitcast<u32>(gidx1)<<2u));
  var alu0 = (gidx0+cast0);
  var alu1 = (gidx1<144);
  var val0 = select(0.0f, data1_576[alu0], alu1);
  var val1 = select(0.0f, data1_576[((cast0-gidx0)+3)], alu1);
  var alu2 = select(0.0f,1.0f,alu1);
  var alu3 = select((f32(-INFINITY)),0.0f,(alu2!=0.0f));
  data0_1024[alu0] = (val0+alu3);
  data0_1024[(alu0+2)] = (val1+alu3);
}`;

const r_576_32_18 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<i32,32>;
@group(0) @binding(1)var<storage,read_write>data0_576:array<i32>;
@group(0) @binding(2)var<storage,read_write>data1_576:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<i32,1>;
  var acc1: array<i32,1>;
  var gidx0 = i32(gindex.x); /* 576 */
  var val0 = data1_576[gidx0];
  var lidx0 = i32(lindex.x); /* 32 */
  acc0[0] = 0;
  for (var Ridx0 = 0; Ridx0 < 18; Ridx0++) {
    var alu1 = ((lidx0*18)+Ridx0);
    var val1 = data1_576[alu1];
    acc0[0] = (acc0[0]+(i32(((alu1<(gidx0+1))&(val1==val0)))));
  }
  temp0[lidx0] = acc0[0];
  workgroupBarrier();
  acc1[0] = 0;
  for (var Ridx102 = 0; Ridx102 < 32; Ridx102++) {
    var val2 = temp0[Ridx102];
    acc1[0] = (acc1[0]+val2);
  }
  var alu9 = (lidx0==0);
  if (alu9) {
    data0_576[gidx0] = acc1[0];
  }
}`;

const E_512_2 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_1024:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_1024:array<f32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 512 */
  var cast0 = bitcast<i32>((bitcast<u32>(gidx0)<<1u));
  var val0 = data1_1024[cast0];
  var alu0 = (cast0+1);
  var val1 = data1_1024[alu0];
  var alu1 = -val0;
  var alu2 = -val1;
  var alu3 = select(alu1,alu2,(alu1<alu2));
  var alu4 = select(val0,val1,(val0<val1));
  data0_1024[cast0] = alu4;
  data0_1024[alu0] = -alu3;
}`;

const r_576_4_16_16 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32,16>;
@group(0) @binding(1)var<storage,read_write>data0_2304:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_147456:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_1024:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_4:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,1>;
  var acc1: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 576 */
  var lidx0 = i32(lindex.x); /* 16 */
  var cast0 = bitcast<u32>(gidx1);
  acc0[0] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 16; Ridx0++) {
    var alu1 = (bitcast<i32>((bitcast<u32>(lidx0)<<4u))+Ridx0);
    var val0 = data1_147456[(alu1+bitcast<i32>((cast0<<8u)))];
    var val1 = data2_1024[(alu1+bitcast<i32>((bitcast<u32>(gidx0)<<8u)))];
    acc0[0] = (acc0[0]+(val0*val1));
  }
  temp0[lidx0] = acc0[0];
  workgroupBarrier();
  acc1[0] = 0.0f;
  for (var Ridx103 = 0; Ridx103 < 16; Ridx103++) {
    var val2 = temp0[Ridx103];
    acc1[0] = (acc1[0]+val2);
  }
  var val3 = data3_4[gidx0];
  var alu9 = (lidx0==0);
  if (alu9) {
    data0_2304[(gidx0+bitcast<i32>((cast0<<2u)))] = (acc1[0]+val3);
  }
}`;

const E_128_2_2_2 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_1024:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_1024:array<f32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 2 */
  var gidx2 = i32(gindex.z); /* 128 */
  var cast0 = bitcast<u32>(gidx2);
  var cast1 = bitcast<i32>((cast0<<3u));
  var val0 = data1_1024[(gidx0+cast1)];
  var gidx1 = i32(gindex.y); /* 2 */
  var alu0 = -gidx1;
  var val1 = select(0.0f, data1_1024[(gidx0+bitcast<i32>((bitcast<u32>(((gidx1+bitcast<i32>((cast0<<2u))+3)&511))<<1u)))], (-1<alu0));
  var alu1 = (cast1-gidx0);
  var val2 = data1_1024[(alu1+3)];
  var val3 = data1_1024[(alu1+5)];
  var alu2 = (gidx0+bitcast<i32>((bitcast<u32>(gidx1)<<1u))+cast1);
  var alu3 = (gidx1<1);
  var alu4 = select(0.0f,val0,alu3);
  var alu5 = select(val2,0.0f,alu3);
  var alu6 = select(0.0f,val3,(alu0<0));
  data0_1024[alu2] = (alu4+alu5);
  data0_1024[(alu2+4)] = (alu6+val1);
}`;

const E_256_2_2n1 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_1024:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_1024:array<f32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 2 */
  var gidx1 = i32(gindex.y); /* 256 */
  var alu0 = (gidx0+bitcast<i32>((bitcast<u32>(gidx1)<<2u)));
  var val0 = data1_1024[alu0];
  var alu1 = (alu0+2);
  var val1 = data1_1024[alu1];
  var alu2 = -val0;
  var alu3 = -val1;
  var alu4 = select(alu2,alu3,(alu2<alu3));
  var alu5 = select(val0,val1,(val0<val1));
  data0_1024[alu0] = alu5;
  data0_1024[alu1] = -alu4;
}`;

const E_64_2_4_2 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_1024:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_1024:array<f32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx2 = i32(gindex.z); /* 64 */
  var cast0 = bitcast<u32>(gidx2);
  var cast1 = bitcast<i32>((cast0<<4u));
  var alu0 = (gidx0+cast1);
  var val0 = data1_1024[alu0];
  var gidx1 = i32(gindex.y); /* 2 */
  var alu1 = -gidx1;
  var val1 = select(0.0f, data1_1024[(gidx0+bitcast<i32>((bitcast<u32>(((gidx1+bitcast<i32>((cast0<<2u))+3)&255))<<2u)))], (-1<alu1));
  var alu2 = (cast1-gidx0);
  var val2 = data1_1024[(alu2+7)];
  var val3 = data1_1024[(alu2+11)];
  var alu3 = (alu0+bitcast<i32>((bitcast<u32>(gidx1)<<2u)));
  var alu4 = (gidx1<1);
  var alu5 = select(0.0f,val0,alu4);
  var alu6 = select(val2,0.0f,alu4);
  var alu7 = select(0.0f,val3,(alu1<0));
  data0_1024[alu3] = (alu5+alu6);
  data0_1024[(alu3+8)] = (alu7+val1);
}`;

const E_128_4_2 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_1024:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_1024:array<f32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 128 */
  var alu0 = (gidx0+bitcast<i32>((bitcast<u32>(gidx1)<<3u)));
  var val0 = data1_1024[alu0];
  var alu1 = (alu0+4);
  var val1 = data1_1024[alu1];
  var alu2 = -val0;
  var alu3 = -val1;
  var alu4 = select(alu2,alu3,(alu2<alu3));
  var alu5 = select(val0,val1,(val0<val1));
  data0_1024[alu0] = alu5;
  data0_1024[alu1] = -alu4;
}`;

const E_32_2_8_2 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_1024:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_1024:array<f32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 8 */
  var gidx2 = i32(gindex.z); /* 32 */
  var cast0 = bitcast<u32>(gidx2);
  var cast1 = bitcast<i32>((cast0<<5u));
  var alu0 = (gidx0+cast1);
  var val0 = data1_1024[alu0];
  var gidx1 = i32(gindex.y); /* 2 */
  var alu1 = -gidx1;
  var val1 = select(0.0f, data1_1024[(gidx0+bitcast<i32>((bitcast<u32>(((gidx1+bitcast<i32>((cast0<<2u))+3)&127))<<3u)))], (-1<alu1));
  var alu2 = (cast1-gidx0);
  var val2 = data1_1024[(alu2+15)];
  var val3 = data1_1024[(alu2+23)];
  var alu3 = (alu0+bitcast<i32>((bitcast<u32>(gidx1)<<3u)));
  var alu4 = (gidx1<1);
  var alu5 = select(0.0f,val0,alu4);
  var alu6 = select(val2,0.0f,alu4);
  var alu7 = select(0.0f,val3,(alu1<0));
  data0_1024[alu3] = (alu5+alu6);
  data0_1024[(alu3+16)] = (alu7+val1);
}`;

const E_64_8_2 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_1024:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_1024:array<f32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 8 */
  var gidx1 = i32(gindex.y); /* 64 */
  var alu0 = (gidx0+bitcast<i32>((bitcast<u32>(gidx1)<<4u)));
  var val0 = data1_1024[alu0];
  var alu1 = (alu0+8);
  var val1 = data1_1024[alu1];
  var alu2 = -val0;
  var alu3 = -val1;
  var alu4 = select(alu2,alu3,(alu2<alu3));
  var alu5 = select(val0,val1,(val0<val1));
  data0_1024[alu0] = alu5;
  data0_1024[alu1] = -alu4;
}`;

const E_2_2_16_16 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_1024:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_1024:array<f32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 16 */
  var val0 = data1_1024[gidx0];
  var val1 = data1_1024[(gidx0+64)];
  var val2 = data1_1024[(gidx0+128)];
  var val3 = data1_1024[(gidx0+192)];
  var val4 = data1_1024[(gidx0+256)];
  var val5 = data1_1024[(gidx0+320)];
  var val6 = data1_1024[(gidx0+384)];
  var val7 = data1_1024[(gidx0+448)];
  var val8 = data1_1024[(gidx0+512)];
  var val9 = data1_1024[(gidx0+576)];
  var val10 = data1_1024[(gidx0+640)];
  var val11 = data1_1024[(gidx0+704)];
  var val12 = data1_1024[(gidx0+768)];
  var val13 = data1_1024[(gidx0+832)];
  var val14 = data1_1024[(gidx0+896)];
  var val15 = data1_1024[(gidx0+960)];
  var gidx1 = i32(gindex.y); /* 2 */
  var alu0 = -gidx1;
  var alu1 = (gidx0+bitcast<i32>((bitcast<u32>(gidx1)<<4u)));
  var alu2 = (-1<alu0);
  var val16 = select(0.0f, data1_1024[(alu1+48)], alu2);
  var val17 = select(0.0f, data1_1024[(alu1+112)], alu2);
  var val18 = select(0.0f, data1_1024[(alu1+176)], alu2);
  var val19 = select(0.0f, data1_1024[(alu1+240)], alu2);
  var val20 = select(0.0f, data1_1024[(alu1+304)], alu2);
  var val21 = select(0.0f, data1_1024[(alu1+368)], alu2);
  var val22 = select(0.0f, data1_1024[(alu1+432)], alu2);
  var val23 = select(0.0f, data1_1024[(alu1+496)], alu2);
  var val24 = select(0.0f, data1_1024[(alu1+560)], alu2);
  var val25 = select(0.0f, data1_1024[(alu1+624)], alu2);
  var val26 = select(0.0f, data1_1024[(alu1+688)], alu2);
  var val27 = select(0.0f, data1_1024[(alu1+752)], alu2);
  var val28 = select(0.0f, data1_1024[(alu1+816)], alu2);
  var val29 = select(0.0f, data1_1024[(alu1+880)], alu2);
  var val30 = select(0.0f, data1_1024[(alu1+944)], alu2);
  var val31 = select(0.0f, data1_1024[(gidx0+(gidx1*-1008)+1008)], alu2);
  var val32 = data1_1024[(31-gidx0)];
  var val33 = data1_1024[(47-gidx0)];
  var val34 = data1_1024[(95-gidx0)];
  var val35 = data1_1024[(111-gidx0)];
  var val36 = data1_1024[(159-gidx0)];
  var val37 = data1_1024[(175-gidx0)];
  var val38 = data1_1024[(223-gidx0)];
  var val39 = data1_1024[(239-gidx0)];
  var val40 = data1_1024[(287-gidx0)];
  var val41 = data1_1024[(303-gidx0)];
  var val42 = data1_1024[(351-gidx0)];
  var val43 = data1_1024[(367-gidx0)];
  var val44 = data1_1024[(415-gidx0)];
  var val45 = data1_1024[(431-gidx0)];
  var val46 = data1_1024[(479-gidx0)];
  var val47 = data1_1024[(495-gidx0)];
  var val48 = data1_1024[(543-gidx0)];
  var val49 = data1_1024[(559-gidx0)];
  var val50 = data1_1024[(607-gidx0)];
  var val51 = data1_1024[(623-gidx0)];
  var val52 = data1_1024[(671-gidx0)];
  var val53 = data1_1024[(687-gidx0)];
  var val54 = data1_1024[(735-gidx0)];
  var val55 = data1_1024[(751-gidx0)];
  var val56 = data1_1024[(799-gidx0)];
  var val57 = data1_1024[(815-gidx0)];
  var val58 = data1_1024[(863-gidx0)];
  var val59 = data1_1024[(879-gidx0)];
  var val60 = data1_1024[(927-gidx0)];
  var val61 = data1_1024[(943-gidx0)];
  var val62 = data1_1024[(991-gidx0)];
  var val63 = data1_1024[(1007-gidx0)];
  var gidx2 = i32(gindex.z); /* 2 */
  var alu3 = (alu1+bitcast<i32>((bitcast<u32>(gidx2)<<5u)));
  var alu4 = (gidx1<1);
  var alu5 = select(0.0f,val0,alu4);
  var alu6 = select(val32,0.0f,alu4);
  var alu7 = select(0.0f,val1,alu4);
  var alu8 = select(val34,0.0f,alu4);
  var alu9 = select(0.0f,val2,alu4);
  var alu10 = select(val36,0.0f,alu4);
  var alu11 = select(0.0f,val3,alu4);
  var alu12 = select(val38,0.0f,alu4);
  var alu13 = select(0.0f,val4,alu4);
  var alu14 = select(val40,0.0f,alu4);
  var alu15 = select(0.0f,val5,alu4);
  var alu16 = select(val42,0.0f,alu4);
  var alu17 = select(0.0f,val6,alu4);
  var alu18 = select(val44,0.0f,alu4);
  var alu19 = select(0.0f,val7,alu4);
  var alu20 = select(val46,0.0f,alu4);
  var alu21 = select(0.0f,val8,alu4);
  var alu22 = select(val48,0.0f,alu4);
  var alu23 = select(0.0f,val9,alu4);
  var alu24 = select(val50,0.0f,alu4);
  var alu25 = select(0.0f,val10,alu4);
  var alu26 = select(val52,0.0f,alu4);
  var alu27 = select(0.0f,val11,alu4);
  var alu28 = select(val54,0.0f,alu4);
  var alu29 = select(0.0f,val12,alu4);
  var alu30 = select(val56,0.0f,alu4);
  var alu31 = select(0.0f,val13,alu4);
  var alu32 = select(val58,0.0f,alu4);
  var alu33 = select(0.0f,val14,alu4);
  var alu34 = select(val60,0.0f,alu4);
  var alu35 = select(0.0f,val15,alu4);
  var alu36 = select(val62,0.0f,alu4);
  var alu37 = (alu0<0);
  var alu38 = select(0.0f,val33,alu37);
  var alu39 = (gidx2<1);
  var alu40 = select(0.0f,(alu5+alu6),alu39);
  var alu41 = select((alu38+val16),0.0f,alu39);
  var alu42 = select(0.0f,val35,alu37);
  var alu43 = select(0.0f,(alu7+alu8),alu39);
  var alu44 = select((alu42+val17),0.0f,alu39);
  var alu45 = select(0.0f,val37,alu37);
  var alu46 = select(0.0f,(alu9+alu10),alu39);
  var alu47 = select((alu45+val18),0.0f,alu39);
  var alu48 = select(0.0f,val39,alu37);
  var alu49 = select(0.0f,(alu11+alu12),alu39);
  var alu50 = select((alu48+val19),0.0f,alu39);
  var alu51 = select(0.0f,val41,alu37);
  var alu52 = select(0.0f,(alu13+alu14),alu39);
  var alu53 = select((alu51+val20),0.0f,alu39);
  var alu54 = select(0.0f,val43,alu37);
  var alu55 = select(0.0f,(alu15+alu16),alu39);
  var alu56 = select((alu54+val21),0.0f,alu39);
  var alu57 = select(0.0f,val45,alu37);
  var alu58 = select(0.0f,(alu17+alu18),alu39);
  var alu59 = select((alu57+val22),0.0f,alu39);
  var alu60 = select(0.0f,val47,alu37);
  var alu61 = select(0.0f,(alu19+alu20),alu39);
  var alu62 = select((alu60+val23),0.0f,alu39);
  var alu63 = select(0.0f,val49,alu37);
  var alu64 = select(0.0f,(alu21+alu22),alu39);
  var alu65 = select((alu63+val24),0.0f,alu39);
  var alu66 = select(0.0f,val51,alu37);
  var alu67 = select(0.0f,(alu23+alu24),alu39);
  var alu68 = select((alu66+val25),0.0f,alu39);
  var alu69 = select(0.0f,val53,alu37);
  var alu70 = select(0.0f,(alu25+alu26),alu39);
  var alu71 = select((alu69+val26),0.0f,alu39);
  var alu72 = select(0.0f,val55,alu37);
  var alu73 = select(0.0f,(alu27+alu28),alu39);
  var alu74 = select((alu72+val27),0.0f,alu39);
  var alu75 = select(0.0f,val57,alu37);
  var alu76 = select(0.0f,(alu29+alu30),alu39);
  var alu77 = select((alu75+val28),0.0f,alu39);
  var alu78 = select(0.0f,val59,alu37);
  var alu79 = select(0.0f,(alu31+alu32),alu39);
  var alu80 = select((alu78+val29),0.0f,alu39);
  var alu81 = select(0.0f,val61,alu37);
  var alu82 = select(0.0f,(alu33+alu34),alu39);
  var alu83 = select((alu81+val30),0.0f,alu39);
  var alu84 = select(0.0f,val63,alu37);
  var alu85 = select(0.0f,(alu35+alu36),alu39);
  var alu86 = select((alu84+val31),0.0f,alu39);
  data0_1024[alu3] = (alu40+alu41);
  data0_1024[(alu3+64)] = (alu43+alu44);
  data0_1024[(alu3+128)] = (alu46+alu47);
  data0_1024[(alu3+192)] = (alu49+alu50);
  data0_1024[(alu3+256)] = (alu52+alu53);
  data0_1024[(alu3+320)] = (alu55+alu56);
  data0_1024[(alu3+384)] = (alu58+alu59);
  data0_1024[(alu3+448)] = (alu61+alu62);
  data0_1024[(alu3+512)] = (alu64+alu65);
  data0_1024[(alu3+576)] = (alu67+alu68);
  data0_1024[(alu3+640)] = (alu70+alu71);
  data0_1024[(alu3+704)] = (alu73+alu74);
  data0_1024[(alu3+768)] = (alu76+alu77);
  data0_1024[(alu3+832)] = (alu79+alu80);
  data0_1024[(alu3+896)] = (alu82+alu83);
  data0_1024[(alu3+960)] = (alu85+alu86);
}`;

const E_32_16_2 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_1024:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_1024:array<f32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 32 */
  var alu0 = (gidx0+bitcast<i32>((bitcast<u32>(gidx1)<<5u)));
  var val0 = data1_1024[alu0];
  var alu1 = (alu0+16);
  var val1 = data1_1024[alu1];
  var alu2 = -val0;
  var alu3 = -val1;
  var alu4 = select(alu2,alu3,(alu2<alu3));
  var alu5 = select(val0,val1,(val0<val1));
  data0_1024[alu0] = alu5;
  data0_1024[alu1] = -alu4;
}`;

const E_2_2_32_8 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_1024:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_1024:array<f32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 32 */
  var val0 = data1_1024[gidx0];
  var val1 = data1_1024[(gidx0+128)];
  var val2 = data1_1024[(gidx0+256)];
  var val3 = data1_1024[(gidx0+384)];
  var val4 = data1_1024[(gidx0+512)];
  var val5 = data1_1024[(gidx0+640)];
  var val6 = data1_1024[(gidx0+768)];
  var val7 = data1_1024[(gidx0+896)];
  var gidx1 = i32(gindex.y); /* 2 */
  var alu0 = -gidx1;
  var alu1 = (gidx0+bitcast<i32>((bitcast<u32>(gidx1)<<5u)));
  var alu2 = (-1<alu0);
  var val8 = select(0.0f, data1_1024[(alu1+96)], alu2);
  var val9 = select(0.0f, data1_1024[(alu1+224)], alu2);
  var val10 = select(0.0f, data1_1024[(alu1+352)], alu2);
  var val11 = select(0.0f, data1_1024[(alu1+480)], alu2);
  var val12 = select(0.0f, data1_1024[(alu1+608)], alu2);
  var val13 = select(0.0f, data1_1024[(alu1+736)], alu2);
  var val14 = select(0.0f, data1_1024[(alu1+864)], alu2);
  var val15 = select(0.0f, data1_1024[(gidx0+(gidx1*-992)+992)], alu2);
  var val16 = data1_1024[(63-gidx0)];
  var val17 = data1_1024[(95-gidx0)];
  var val18 = data1_1024[(191-gidx0)];
  var val19 = data1_1024[(223-gidx0)];
  var val20 = data1_1024[(319-gidx0)];
  var val21 = data1_1024[(351-gidx0)];
  var val22 = data1_1024[(447-gidx0)];
  var val23 = data1_1024[(479-gidx0)];
  var val24 = data1_1024[(575-gidx0)];
  var val25 = data1_1024[(607-gidx0)];
  var val26 = data1_1024[(703-gidx0)];
  var val27 = data1_1024[(735-gidx0)];
  var val28 = data1_1024[(831-gidx0)];
  var val29 = data1_1024[(863-gidx0)];
  var val30 = data1_1024[(959-gidx0)];
  var val31 = data1_1024[(991-gidx0)];
  var gidx2 = i32(gindex.z); /* 2 */
  var alu3 = (alu1+bitcast<i32>((bitcast<u32>(gidx2)<<6u)));
  var alu4 = (gidx1<1);
  var alu5 = select(0.0f,val0,alu4);
  var alu6 = select(val16,0.0f,alu4);
  var alu7 = select(0.0f,val1,alu4);
  var alu8 = select(val18,0.0f,alu4);
  var alu9 = select(0.0f,val2,alu4);
  var alu10 = select(val20,0.0f,alu4);
  var alu11 = select(0.0f,val3,alu4);
  var alu12 = select(val22,0.0f,alu4);
  var alu13 = select(0.0f,val4,alu4);
  var alu14 = select(val24,0.0f,alu4);
  var alu15 = select(0.0f,val5,alu4);
  var alu16 = select(val26,0.0f,alu4);
  var alu17 = select(0.0f,val6,alu4);
  var alu18 = select(val28,0.0f,alu4);
  var alu19 = select(0.0f,val7,alu4);
  var alu20 = select(val30,0.0f,alu4);
  var alu21 = (alu0<0);
  var alu22 = select(0.0f,val17,alu21);
  var alu23 = (gidx2<1);
  var alu24 = select(0.0f,(alu5+alu6),alu23);
  var alu25 = select((alu22+val8),0.0f,alu23);
  var alu26 = select(0.0f,val19,alu21);
  var alu27 = select(0.0f,(alu7+alu8),alu23);
  var alu28 = select((alu26+val9),0.0f,alu23);
  var alu29 = select(0.0f,val21,alu21);
  var alu30 = select(0.0f,(alu9+alu10),alu23);
  var alu31 = select((alu29+val10),0.0f,alu23);
  var alu32 = select(0.0f,val23,alu21);
  var alu33 = select(0.0f,(alu11+alu12),alu23);
  var alu34 = select((alu32+val11),0.0f,alu23);
  var alu35 = select(0.0f,val25,alu21);
  var alu36 = select(0.0f,(alu13+alu14),alu23);
  var alu37 = select((alu35+val12),0.0f,alu23);
  var alu38 = select(0.0f,val27,alu21);
  var alu39 = select(0.0f,(alu15+alu16),alu23);
  var alu40 = select((alu38+val13),0.0f,alu23);
  var alu41 = select(0.0f,val29,alu21);
  var alu42 = select(0.0f,(alu17+alu18),alu23);
  var alu43 = select((alu41+val14),0.0f,alu23);
  var alu44 = select(0.0f,val31,alu21);
  var alu45 = select(0.0f,(alu19+alu20),alu23);
  var alu46 = select((alu44+val15),0.0f,alu23);
  data0_1024[alu3] = (alu24+alu25);
  data0_1024[(alu3+128)] = (alu27+alu28);
  data0_1024[(alu3+256)] = (alu30+alu31);
  data0_1024[(alu3+384)] = (alu33+alu34);
  data0_1024[(alu3+512)] = (alu36+alu37);
  data0_1024[(alu3+640)] = (alu39+alu40);
  data0_1024[(alu3+768)] = (alu42+alu43);
  data0_1024[(alu3+896)] = (alu45+alu46);
}`;

const E_2_32_16 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_1024:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_1024:array<f32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 32 */
  var val0 = data1_1024[gidx0];
  var val1 = data1_1024[(gidx0+32)];
  var val2 = data1_1024[(gidx0+64)];
  var val3 = data1_1024[(gidx0+96)];
  var val4 = data1_1024[(gidx0+128)];
  var val5 = data1_1024[(gidx0+160)];
  var val6 = data1_1024[(gidx0+192)];
  var val7 = data1_1024[(gidx0+224)];
  var val8 = data1_1024[(gidx0+256)];
  var val9 = data1_1024[(gidx0+288)];
  var val10 = data1_1024[(gidx0+320)];
  var val11 = data1_1024[(gidx0+352)];
  var val12 = data1_1024[(gidx0+384)];
  var val13 = data1_1024[(gidx0+416)];
  var val14 = data1_1024[(gidx0+448)];
  var val15 = data1_1024[(gidx0+480)];
  var val16 = data1_1024[(gidx0+512)];
  var val17 = data1_1024[(gidx0+544)];
  var val18 = data1_1024[(gidx0+576)];
  var val19 = data1_1024[(gidx0+608)];
  var val20 = data1_1024[(gidx0+640)];
  var val21 = data1_1024[(gidx0+672)];
  var val22 = data1_1024[(gidx0+704)];
  var val23 = data1_1024[(gidx0+736)];
  var val24 = data1_1024[(gidx0+768)];
  var val25 = data1_1024[(gidx0+800)];
  var val26 = data1_1024[(gidx0+832)];
  var val27 = data1_1024[(gidx0+864)];
  var val28 = data1_1024[(gidx0+896)];
  var val29 = data1_1024[(gidx0+928)];
  var val30 = data1_1024[(gidx0+960)];
  var val31 = data1_1024[(gidx0+992)];
  var gidx1 = i32(gindex.y); /* 2 */
  var alu0 = (gidx0+bitcast<i32>((bitcast<u32>(gidx1)<<5u)));
  var alu1 = -val0;
  var alu2 = -val1;
  var alu3 = -val2;
  var alu4 = -val3;
  var alu5 = -val4;
  var alu6 = -val5;
  var alu7 = -val6;
  var alu8 = -val7;
  var alu9 = -val8;
  var alu10 = -val9;
  var alu11 = -val10;
  var alu12 = -val11;
  var alu13 = -val12;
  var alu14 = -val13;
  var alu15 = -val14;
  var alu16 = -val15;
  var alu17 = -val16;
  var alu18 = -val17;
  var alu19 = -val18;
  var alu20 = -val19;
  var alu21 = -val20;
  var alu22 = -val21;
  var alu23 = -val22;
  var alu24 = -val23;
  var alu25 = -val24;
  var alu26 = -val25;
  var alu27 = -val26;
  var alu28 = -val27;
  var alu29 = -val28;
  var alu30 = -val29;
  var alu31 = -val30;
  var alu32 = -val31;
  var alu33 = select(alu1,alu2,(alu1<alu2));
  var alu34 = select(alu3,alu4,(alu3<alu4));
  var alu35 = select(alu5,alu6,(alu5<alu6));
  var alu36 = select(alu7,alu8,(alu7<alu8));
  var alu37 = select(alu9,alu10,(alu9<alu10));
  var alu38 = select(alu11,alu12,(alu11<alu12));
  var alu39 = select(alu13,alu14,(alu13<alu14));
  var alu40 = select(alu15,alu16,(alu15<alu16));
  var alu41 = select(alu17,alu18,(alu17<alu18));
  var alu42 = select(alu19,alu20,(alu19<alu20));
  var alu43 = select(alu21,alu22,(alu21<alu22));
  var alu44 = select(alu23,alu24,(alu23<alu24));
  var alu45 = select(alu25,alu26,(alu25<alu26));
  var alu46 = select(alu27,alu28,(alu27<alu28));
  var alu47 = select(alu29,alu30,(alu29<alu30));
  var alu48 = select(alu31,alu32,(alu31<alu32));
  var alu49 = (gidx1<1);
  var alu50 = select(val0,val1,(val0<val1));
  var alu51 = select(0.0f,alu50,alu49);
  var alu52 = select(-alu33,0.0f,alu49);
  var alu53 = select(val2,val3,(val2<val3));
  var alu54 = select(0.0f,alu53,alu49);
  var alu55 = select(-alu34,0.0f,alu49);
  var alu56 = select(val4,val5,(val4<val5));
  var alu57 = select(0.0f,alu56,alu49);
  var alu58 = select(-alu35,0.0f,alu49);
  var alu59 = select(val6,val7,(val6<val7));
  var alu60 = select(0.0f,alu59,alu49);
  var alu61 = select(-alu36,0.0f,alu49);
  var alu62 = select(val8,val9,(val8<val9));
  var alu63 = select(0.0f,alu62,alu49);
  var alu64 = select(-alu37,0.0f,alu49);
  var alu65 = select(val10,val11,(val10<val11));
  var alu66 = select(0.0f,alu65,alu49);
  var alu67 = select(-alu38,0.0f,alu49);
  var alu68 = select(val12,val13,(val12<val13));
  var alu69 = select(0.0f,alu68,alu49);
  var alu70 = select(-alu39,0.0f,alu49);
  var alu71 = select(val14,val15,(val14<val15));
  var alu72 = select(0.0f,alu71,alu49);
  var alu73 = select(-alu40,0.0f,alu49);
  var alu74 = select(val16,val17,(val16<val17));
  var alu75 = select(0.0f,alu74,alu49);
  var alu76 = select(-alu41,0.0f,alu49);
  var alu77 = select(val18,val19,(val18<val19));
  var alu78 = select(0.0f,alu77,alu49);
  var alu79 = select(-alu42,0.0f,alu49);
  var alu80 = select(val20,val21,(val20<val21));
  var alu81 = select(0.0f,alu80,alu49);
  var alu82 = select(-alu43,0.0f,alu49);
  var alu83 = select(val22,val23,(val22<val23));
  var alu84 = select(0.0f,alu83,alu49);
  var alu85 = select(-alu44,0.0f,alu49);
  var alu86 = select(val24,val25,(val24<val25));
  var alu87 = select(0.0f,alu86,alu49);
  var alu88 = select(-alu45,0.0f,alu49);
  var alu89 = select(val26,val27,(val26<val27));
  var alu90 = select(0.0f,alu89,alu49);
  var alu91 = select(-alu46,0.0f,alu49);
  var alu92 = select(val28,val29,(val28<val29));
  var alu93 = select(0.0f,alu92,alu49);
  var alu94 = select(-alu47,0.0f,alu49);
  var alu95 = select(val30,val31,(val30<val31));
  var alu96 = select(0.0f,alu95,alu49);
  var alu97 = select(-alu48,0.0f,alu49);
  data0_1024[alu0] = (alu51+alu52);
  data0_1024[(alu0+64)] = (alu54+alu55);
  data0_1024[(alu0+128)] = (alu57+alu58);
  data0_1024[(alu0+192)] = (alu60+alu61);
  data0_1024[(alu0+256)] = (alu63+alu64);
  data0_1024[(alu0+320)] = (alu66+alu67);
  data0_1024[(alu0+384)] = (alu69+alu70);
  data0_1024[(alu0+448)] = (alu72+alu73);
  data0_1024[(alu0+512)] = (alu75+alu76);
  data0_1024[(alu0+576)] = (alu78+alu79);
  data0_1024[(alu0+640)] = (alu81+alu82);
  data0_1024[(alu0+704)] = (alu84+alu85);
  data0_1024[(alu0+768)] = (alu87+alu88);
  data0_1024[(alu0+832)] = (alu90+alu91);
  data0_1024[(alu0+896)] = (alu93+alu94);
  data0_1024[(alu0+960)] = (alu96+alu97);
}`;

const E_2_2_64_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_1024:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_1024:array<f32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 64 */
  var val0 = data1_1024[gidx0];
  var val1 = data1_1024[(gidx0+256)];
  var val2 = data1_1024[(gidx0+512)];
  var val3 = data1_1024[(gidx0+768)];
  var gidx1 = i32(gindex.y); /* 2 */
  var alu0 = -gidx1;
  var alu1 = (gidx0+bitcast<i32>((bitcast<u32>(gidx1)<<6u)));
  var alu2 = (-1<alu0);
  var val4 = select(0.0f, data1_1024[(alu1+192)], alu2);
  var val5 = select(0.0f, data1_1024[(alu1+448)], alu2);
  var val6 = select(0.0f, data1_1024[(alu1+704)], alu2);
  var val7 = select(0.0f, data1_1024[(gidx0+(gidx1*-960)+960)], alu2);
  var val8 = data1_1024[(127-gidx0)];
  var val9 = data1_1024[(191-gidx0)];
  var val10 = data1_1024[(383-gidx0)];
  var val11 = data1_1024[(447-gidx0)];
  var val12 = data1_1024[(639-gidx0)];
  var val13 = data1_1024[(703-gidx0)];
  var val14 = data1_1024[(895-gidx0)];
  var val15 = data1_1024[(959-gidx0)];
  var gidx2 = i32(gindex.z); /* 2 */
  var alu3 = (alu1+bitcast<i32>((bitcast<u32>(gidx2)<<7u)));
  var alu4 = (gidx1<1);
  var alu5 = select(0.0f,val0,alu4);
  var alu6 = select(val8,0.0f,alu4);
  var alu7 = select(0.0f,val1,alu4);
  var alu8 = select(val10,0.0f,alu4);
  var alu9 = select(0.0f,val2,alu4);
  var alu10 = select(val12,0.0f,alu4);
  var alu11 = select(0.0f,val3,alu4);
  var alu12 = select(val14,0.0f,alu4);
  var alu13 = (alu0<0);
  var alu14 = select(0.0f,val9,alu13);
  var alu15 = (gidx2<1);
  var alu16 = select(0.0f,(alu5+alu6),alu15);
  var alu17 = select((alu14+val4),0.0f,alu15);
  var alu18 = select(0.0f,val11,alu13);
  var alu19 = select(0.0f,(alu7+alu8),alu15);
  var alu20 = select((alu18+val5),0.0f,alu15);
  var alu21 = select(0.0f,val13,alu13);
  var alu22 = select(0.0f,(alu9+alu10),alu15);
  var alu23 = select((alu21+val6),0.0f,alu15);
  var alu24 = select(0.0f,val15,alu13);
  var alu25 = select(0.0f,(alu11+alu12),alu15);
  var alu26 = select((alu24+val7),0.0f,alu15);
  data0_1024[alu3] = (alu16+alu17);
  data0_1024[(alu3+256)] = (alu19+alu20);
  data0_1024[(alu3+512)] = (alu22+alu23);
  data0_1024[(alu3+768)] = (alu25+alu26);
}`;

const E_2_64_8 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_1024:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_1024:array<f32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 64 */
  var val0 = data1_1024[gidx0];
  var val1 = data1_1024[(gidx0+64)];
  var val2 = data1_1024[(gidx0+128)];
  var val3 = data1_1024[(gidx0+192)];
  var val4 = data1_1024[(gidx0+256)];
  var val5 = data1_1024[(gidx0+320)];
  var val6 = data1_1024[(gidx0+384)];
  var val7 = data1_1024[(gidx0+448)];
  var val8 = data1_1024[(gidx0+512)];
  var val9 = data1_1024[(gidx0+576)];
  var val10 = data1_1024[(gidx0+640)];
  var val11 = data1_1024[(gidx0+704)];
  var val12 = data1_1024[(gidx0+768)];
  var val13 = data1_1024[(gidx0+832)];
  var val14 = data1_1024[(gidx0+896)];
  var val15 = data1_1024[(gidx0+960)];
  var gidx1 = i32(gindex.y); /* 2 */
  var alu0 = (gidx0+bitcast<i32>((bitcast<u32>(gidx1)<<6u)));
  var alu1 = -val0;
  var alu2 = -val1;
  var alu3 = -val2;
  var alu4 = -val3;
  var alu5 = -val4;
  var alu6 = -val5;
  var alu7 = -val6;
  var alu8 = -val7;
  var alu9 = -val8;
  var alu10 = -val9;
  var alu11 = -val10;
  var alu12 = -val11;
  var alu13 = -val12;
  var alu14 = -val13;
  var alu15 = -val14;
  var alu16 = -val15;
  var alu17 = select(alu1,alu2,(alu1<alu2));
  var alu18 = select(alu3,alu4,(alu3<alu4));
  var alu19 = select(alu5,alu6,(alu5<alu6));
  var alu20 = select(alu7,alu8,(alu7<alu8));
  var alu21 = select(alu9,alu10,(alu9<alu10));
  var alu22 = select(alu11,alu12,(alu11<alu12));
  var alu23 = select(alu13,alu14,(alu13<alu14));
  var alu24 = select(alu15,alu16,(alu15<alu16));
  var alu25 = (gidx1<1);
  var alu26 = select(val0,val1,(val0<val1));
  var alu27 = select(0.0f,alu26,alu25);
  var alu28 = select(-alu17,0.0f,alu25);
  var alu29 = select(val2,val3,(val2<val3));
  var alu30 = select(0.0f,alu29,alu25);
  var alu31 = select(-alu18,0.0f,alu25);
  var alu32 = select(val4,val5,(val4<val5));
  var alu33 = select(0.0f,alu32,alu25);
  var alu34 = select(-alu19,0.0f,alu25);
  var alu35 = select(val6,val7,(val6<val7));
  var alu36 = select(0.0f,alu35,alu25);
  var alu37 = select(-alu20,0.0f,alu25);
  var alu38 = select(val8,val9,(val8<val9));
  var alu39 = select(0.0f,alu38,alu25);
  var alu40 = select(-alu21,0.0f,alu25);
  var alu41 = select(val10,val11,(val10<val11));
  var alu42 = select(0.0f,alu41,alu25);
  var alu43 = select(-alu22,0.0f,alu25);
  var alu44 = select(val12,val13,(val12<val13));
  var alu45 = select(0.0f,alu44,alu25);
  var alu46 = select(-alu23,0.0f,alu25);
  var alu47 = select(val14,val15,(val14<val15));
  var alu48 = select(0.0f,alu47,alu25);
  var alu49 = select(-alu24,0.0f,alu25);
  data0_1024[alu0] = (alu27+alu28);
  data0_1024[(alu0+128)] = (alu30+alu31);
  data0_1024[(alu0+256)] = (alu33+alu34);
  data0_1024[(alu0+384)] = (alu36+alu37);
  data0_1024[(alu0+512)] = (alu39+alu40);
  data0_1024[(alu0+640)] = (alu42+alu43);
  data0_1024[(alu0+768)] = (alu45+alu46);
  data0_1024[(alu0+896)] = (alu48+alu49);
}`;

const E_2_2_128_2 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_1024:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_1024:array<f32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 128 */
  var val0 = data1_1024[gidx0];
  var val1 = data1_1024[(gidx0+512)];
  var gidx1 = i32(gindex.y); /* 2 */
  var alu0 = -gidx1;
  var alu1 = (gidx0+bitcast<i32>((bitcast<u32>(gidx1)<<7u)));
  var alu2 = (-1<alu0);
  var val2 = select(0.0f, data1_1024[(alu1+384)], alu2);
  var val3 = select(0.0f, data1_1024[(gidx0+(gidx1*-896)+896)], alu2);
  var val4 = data1_1024[(255-gidx0)];
  var val5 = data1_1024[(383-gidx0)];
  var val6 = data1_1024[(767-gidx0)];
  var val7 = data1_1024[(895-gidx0)];
  var gidx2 = i32(gindex.z); /* 2 */
  var alu3 = (alu1+bitcast<i32>((bitcast<u32>(gidx2)<<8u)));
  var alu4 = (gidx1<1);
  var alu5 = select(0.0f,val0,alu4);
  var alu6 = select(val4,0.0f,alu4);
  var alu7 = select(0.0f,val1,alu4);
  var alu8 = select(val6,0.0f,alu4);
  var alu9 = (alu0<0);
  var alu10 = select(0.0f,val5,alu9);
  var alu11 = (gidx2<1);
  var alu12 = select(0.0f,(alu5+alu6),alu11);
  var alu13 = select((alu10+val2),0.0f,alu11);
  var alu14 = select(0.0f,val7,alu9);
  var alu15 = select(0.0f,(alu7+alu8),alu11);
  var alu16 = select((alu14+val3),0.0f,alu11);
  data0_1024[alu3] = (alu12+alu13);
  data0_1024[(alu3+512)] = (alu15+alu16);
}`;

const E_2_128_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_1024:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_1024:array<f32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 128 */
  var val0 = data1_1024[gidx0];
  var val1 = data1_1024[(gidx0+128)];
  var val2 = data1_1024[(gidx0+256)];
  var val3 = data1_1024[(gidx0+384)];
  var val4 = data1_1024[(gidx0+512)];
  var val5 = data1_1024[(gidx0+640)];
  var val6 = data1_1024[(gidx0+768)];
  var val7 = data1_1024[(gidx0+896)];
  var gidx1 = i32(gindex.y); /* 2 */
  var alu0 = (gidx0+bitcast<i32>((bitcast<u32>(gidx1)<<7u)));
  var alu1 = -val0;
  var alu2 = -val1;
  var alu3 = -val2;
  var alu4 = -val3;
  var alu5 = -val4;
  var alu6 = -val5;
  var alu7 = -val6;
  var alu8 = -val7;
  var alu9 = select(alu1,alu2,(alu1<alu2));
  var alu10 = select(alu3,alu4,(alu3<alu4));
  var alu11 = select(alu5,alu6,(alu5<alu6));
  var alu12 = select(alu7,alu8,(alu7<alu8));
  var alu13 = (gidx1<1);
  var alu14 = select(val0,val1,(val0<val1));
  var alu15 = select(0.0f,alu14,alu13);
  var alu16 = select(-alu9,0.0f,alu13);
  var alu17 = select(val2,val3,(val2<val3));
  var alu18 = select(0.0f,alu17,alu13);
  var alu19 = select(-alu10,0.0f,alu13);
  var alu20 = select(val4,val5,(val4<val5));
  var alu21 = select(0.0f,alu20,alu13);
  var alu22 = select(-alu11,0.0f,alu13);
  var alu23 = select(val6,val7,(val6<val7));
  var alu24 = select(0.0f,alu23,alu13);
  var alu25 = select(-alu12,0.0f,alu13);
  data0_1024[alu0] = (alu15+alu16);
  data0_1024[(alu0+256)] = (alu18+alu19);
  data0_1024[(alu0+512)] = (alu21+alu22);
  data0_1024[(alu0+768)] = (alu24+alu25);
}`;

const E_2_256_2 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_1024:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_1024:array<f32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 256 */
  var val0 = data1_1024[gidx0];
  var gidx1 = i32(gindex.y); /* 2 */
  var alu0 = -gidx1;
  var val1 = select(0.0f, data1_1024[(gidx0+(gidx1*-768)+768)], (-1<alu0));
  var val2 = data1_1024[(511-gidx0)];
  var val3 = data1_1024[(767-gidx0)];
  var alu1 = (gidx0+bitcast<i32>((bitcast<u32>(gidx1)<<8u)));
  var alu2 = (gidx1<1);
  var alu3 = select(0.0f,val0,alu2);
  var alu4 = select(val2,0.0f,alu2);
  var alu5 = select(0.0f,val3,(alu0<0));
  data0_1024[alu1] = (alu3+alu4);
  data0_1024[(alu1+512)] = (alu5+val1);
}`;

const E_2_256_2n1 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_1024:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_1024:array<f32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 256 */
  var val0 = data1_1024[gidx0];
  var val1 = data1_1024[(gidx0+256)];
  var val2 = data1_1024[(gidx0+512)];
  var val3 = data1_1024[(gidx0+768)];
  var gidx1 = i32(gindex.y); /* 2 */
  var alu0 = (gidx0+bitcast<i32>((bitcast<u32>(gidx1)<<8u)));
  var alu1 = -val0;
  var alu2 = -val1;
  var alu3 = -val2;
  var alu4 = -val3;
  var alu5 = select(alu1,alu2,(alu1<alu2));
  var alu6 = select(alu3,alu4,(alu3<alu4));
  var alu7 = (gidx1<1);
  var alu8 = select(val0,val1,(val0<val1));
  var alu9 = select(0.0f,alu8,alu7);
  var alu10 = select(-alu5,0.0f,alu7);
  var alu11 = select(val2,val3,(val2<val3));
  var alu12 = select(0.0f,alu11,alu7);
  var alu13 = select(-alu6,0.0f,alu7);
  data0_1024[alu0] = (alu9+alu10);
  data0_1024[(alu0+512)] = (alu12+alu13);
}`;

const E_512_2n1 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_1024:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_1024:array<f32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 512 */
  var val0 = data1_1024[gidx0];
  var val1 = data1_1024[(1023-gidx0)];
  var alu0 = -val0;
  var alu1 = -val1;
  var alu2 = select(alu0,alu1,(alu0<alu1));
  var alu3 = select(val0,val1,(val0<val1));
  data0_1024[gidx0] = alu3;
  data0_1024[(gidx0+512)] = -alu2;
}`;

const r_72_8_576 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_576:array<i32>;
@group(0) @binding(2)var<storage,read_write>data1_1024:array<f32>;
@compute @workgroup_size(8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<i32,1>;
  var gidx0 = i32(gindex.x); /* 72 */
  var lidx0 = i32(lindex.x); /* 8 */
  var alu0 = (lidx0+bitcast<i32>((bitcast<u32>(gidx0)<<3u)));
  var val0 = data1_1024[alu0];
  acc0[0] = 0;
  for (var Ridx0 = 0; Ridx0 < 576; Ridx0++) {
    var val1 = data1_1024[Ridx0];
    acc0[0] = (acc0[0]+(i32(((Ridx0<(alu0+1))&(val1==val0)))));
  }
  data0_576[alu0] = acc0[0];
}`;

const r_144_16_4_36 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<i32,64>;
@group(0) @binding(1)var<storage,read_write>data0_576:array<i32>;
@group(0) @binding(2)var<storage,read_write>data1_576:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_1024:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_576:array<i32>;
@group(0) @binding(5)var<storage,read_write>data4_576:array<i32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<i32,4>;
  var acc1: array<i32,4>;
  var gidx0 = i32(gindex.x); /* 144 */
  var cast0 = bitcast<i32>((bitcast<u32>(gidx0)<<2u));
  var val0 = data4_576[cast0];
  var alu0 = (cast0+1);
  var val1 = data4_576[alu0];
  var alu1 = (cast0+2);
  var val2 = data4_576[alu1];
  var alu2 = (cast0+3);
  var val3 = data4_576[alu2];
  var val4 = data2_1024[cast0];
  var val5 = data2_1024[alu0];
  var val6 = data2_1024[alu1];
  var val7 = data2_1024[alu2];
  var lidx0 = i32(lindex.x); /* 16 */
  acc0[0] = 0;
  acc0[1] = 0;
  acc0[2] = 0;
  acc0[3] = 0;
  for (var Ridx0 = 0; Ridx0 < 36; Ridx0++) {
    var alu7 = ((lidx0*36)+Ridx0);
    var val8 = data3_576[alu7];
    var val9 = data1_576[alu7];
    acc0[0] = (acc0[0]+((i32(((val9==val4)&(val8==val0))))*alu7));
    acc0[1] = (acc0[1]+((i32(((val9==val5)&(val8==val1))))*alu7));
    acc0[2] = (acc0[2]+((i32(((val9==val6)&(val8==val2))))*alu7));
    acc0[3] = (acc0[3]+((i32(((val9==val7)&(val8==val3))))*alu7));
  }
  var cast1 = bitcast<i32>((bitcast<u32>(lidx0)<<2u));
  temp0[cast1] = acc0[0];
  temp0[(cast1+1)] = acc0[1];
  temp0[(cast1+2)] = acc0[2];
  temp0[(cast1+3)] = acc0[3];
  workgroupBarrier();
  acc1[0] = 0;
  acc1[1] = 0;
  acc1[2] = 0;
  acc1[3] = 0;
  for (var Ridx102 = 0; Ridx102 < 16; Ridx102++) {
    var cast2 = bitcast<i32>((bitcast<u32>(Ridx102)<<2u));
    var val10 = temp0[cast2];
    var val11 = temp0[(cast2+1)];
    var val12 = temp0[(cast2+2)];
    var val13 = temp0[(cast2+3)];
    acc1[0] = (acc1[0]+val10);
    acc1[1] = (acc1[1]+val11);
    acc1[2] = (acc1[2]+val12);
    acc1[3] = (acc1[3]+val13);
  }
  var alu27 = (lidx0==0);
  if (alu27) {
    data0_576[cast0] = acc1[0];
  }
  if (alu27) {
    data0_576[alu0] = acc1[1];
  }
  if (alu27) {
    data0_576[alu1] = acc1[2];
  }
  if (alu27) {
    data0_576[alu2] = acc1[3];
  }
}`;

const r_150_24_4_2_24_24_24_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32,192>;
@group(0) @binding(1)var<storage,read_write>data0_1200:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_576:array<i32>;
@group(0) @binding(3)var<storage,read_write>data2_2304:array<f32>;
@compute @workgroup_size(24) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<i32,1>;
  var acc1: array<i32,1>;
  var acc2: array<f32,8>;
  var acc3: array<f32,8>;
  acc0[0] = 0;
  for (var Ridx2 = 0; Ridx2 < 24; Ridx2++) {
    var alu1 = (16.0f*((f32((Ridx2+1)))+-0.5f));
    var alu2 = select((alu1+-0.5f),0.0f,(alu1<0.5f));
    var alu3 = select(alu2,383.0f,(383.0f<alu2));
    var alu4 = trunc(alu3);
    var alu5 = select(alu4,(alu4+-1.0f),(alu3<alu4));
    acc0[0] = (acc0[0]+(i32(((alu3-alu5)!=0.0f))));
  }
  acc1[0] = 0;
  for (var Ridx3 = 0; Ridx3 < 24; Ridx3++) {
    var alu9 = (16.0f*((f32((Ridx3+1)))+-0.5f));
    var alu10 = select((alu9+-0.5f),0.0f,(alu9<0.5f));
    var alu11 = select(alu10,383.0f,(383.0f<alu10));
    var alu12 = trunc(alu11);
    var cast0 = (i32(alu12));
    var alu13 = (alu12+-1.0f);
    var alu14 = select(cast0,(i32((alu12+1.0f))),(alu12<alu11));
    var alu15 = (alu11<alu12);
    var alu16 = select(cast0,(i32(alu13)),alu15);
    var alu17 = select(alu12,alu13,alu15);
    var alu18 = select(0.0f,0.5f,((-1<alu14)&(alu14<384)));
    var alu19 = select(0.0f,0.5f,((-1<alu16)&(alu16<384)));
    acc1[0] = (acc1[0]+(i32(((alu19+((alu18-alu19)*(alu11-alu17)))!=0.0f))));
  }
  var gidx0 = i32(gindex.x); /* 150 */
  var cast1 = bitcast<u32>(gidx0);
  var cast2 = bitcast<i32>((cast1<<1u));
  var val0 = data1_576[cast2];
  var val1 = data1_576[(cast2+1)];
  var lidx0 = i32(lindex.x); /* 24 */
  var alu22 = (16.0f*((f32((lidx0+1)))+-0.5f));
  var alu23 = select((alu22+-0.5f),0.0f,(alu22<0.5f));
  var alu24 = select(alu23,383.0f,(383.0f<alu23));
  var alu25 = trunc(alu24);
  var cast3 = (i32(alu25));
  var alu26 = (alu25+-1.0f);
  var alu27 = (((f32(lidx0))+0.5f)*(1/(f32(acc1[0]))));
  var alu28 = select(cast3,(i32((alu25+1.0f))),(alu25<alu24));
  var alu29 = (alu24<alu25);
  var alu30 = select(cast3,(i32(alu26)),alu29);
  var alu31 = select(alu25,alu26,alu29);
  acc2[0] = 0.0f;
  acc2[1] = 0.0f;
  acc2[2] = 0.0f;
  acc2[3] = 0.0f;
  acc2[4] = 0.0f;
  acc2[5] = 0.0f;
  acc2[6] = 0.0f;
  acc2[7] = 0.0f;
  for (var Ridx5_1 = 0; Ridx5_1 < 24; Ridx5_1++) {
    var alu40 = ((lidx0*96)+bitcast<i32>((bitcast<u32>(Ridx5_1)<<2u)));
    var val2 = data2_2304[alu40];
    var val3 = data2_2304[(alu40+1)];
    var val4 = data2_2304[(alu40+2)];
    var val5 = data2_2304[(alu40+3)];
    var alu41 = (16.0f*((f32((Ridx5_1+1)))+-0.5f));
    var alu42 = select((alu41+-0.5f),0.0f,(alu41<0.5f));
    var alu43 = select(alu42,383.0f,(383.0f<alu42));
    var alu44 = trunc(alu43);
    var alu45 = ((lidx0*24)+Ridx5_1);
    var alu46 = select(alu44,(alu44+-1.0f),(alu43<alu44));
    var alu47 = (alu43-alu46);
    var alu48 = select(0.0f,alu47,((-1<alu28)&(alu28<384)));
    var alu49 = select(0.0f,alu47,((-1<alu30)&(alu30<384)));
    var alu50 = (((f32(Ridx5_1))+0.5f)*(1/(f32(acc0[0]))));
    var alu51 = ((alu49+((alu48-alu49)*(alu24-alu31)))!=0.0f);
    var alu52 = ((((0.01f<alu50)&(alu50<0.99f))!=true)|(((0.01f<alu27)&(alu27<0.99f))!=true));
    var alu53 = select(0.0f,0.05f,alu51);
    var alu54 = select(alu53,0.0f,alu52);
    var alu55 = select(0.0f,alu50,alu51);
    var alu56 = select(alu55,0.0f,alu52);
    var alu57 = ((val2*alu54)+alu56);
    var alu58 = (val0!=alu45);
    var alu59 = select(alu57,0.0f,alu58);
    var alu60 = (val1!=alu45);
    var alu61 = select(alu57,0.0f,alu60);
    var alu62 = select(0.0f,alu27,alu51);
    var alu63 = select(alu62,0.0f,alu52);
    var alu64 = ((val3*alu54)+alu63);
    var alu65 = select(alu64,0.0f,alu58);
    var alu66 = select(alu64,0.0f,alu60);
    var alu67 = (exp2((val4*1.4426950408889634f))*alu54);
    var alu68 = select(alu67,0.0f,alu58);
    var alu69 = select(alu67,0.0f,alu60);
    var alu70 = (exp2((val5*1.4426950408889634f))*alu54);
    var alu71 = select(alu70,0.0f,alu58);
    var alu72 = select(alu70,0.0f,alu60);
    acc2[0] = (acc2[0]+alu59);
    acc2[1] = (acc2[1]+alu61);
    acc2[2] = (acc2[2]+alu65);
    acc2[3] = (acc2[3]+alu66);
    acc2[4] = (acc2[4]+alu68);
    acc2[5] = (acc2[5]+alu69);
    acc2[6] = (acc2[6]+alu71);
    acc2[7] = (acc2[7]+alu72);
  }
  var cast4 = bitcast<i32>((bitcast<u32>(lidx0)<<3u));
  temp0[cast4] = acc2[0];
  temp0[(cast4+1)] = acc2[1];
  temp0[(cast4+2)] = acc2[2];
  temp0[(cast4+3)] = acc2[3];
  temp0[(cast4+4)] = acc2[4];
  temp0[(cast4+5)] = acc2[5];
  temp0[(cast4+6)] = acc2[6];
  temp0[(cast4+7)] = acc2[7];
  workgroupBarrier();
  acc3[0] = 0.0f;
  acc3[1] = 0.0f;
  acc3[2] = 0.0f;
  acc3[3] = 0.0f;
  acc3[4] = 0.0f;
  acc3[5] = 0.0f;
  acc3[6] = 0.0f;
  acc3[7] = 0.0f;
  for (var Ridx109 = 0; Ridx109 < 24; Ridx109++) {
    var cast5 = bitcast<i32>((bitcast<u32>(Ridx109)<<3u));
    var val6 = temp0[cast5];
    var val7 = temp0[(cast5+1)];
    var val8 = temp0[(cast5+2)];
    var val9 = temp0[(cast5+3)];
    var val10 = temp0[(cast5+4)];
    var val11 = temp0[(cast5+5)];
    var val12 = temp0[(cast5+6)];
    var val13 = temp0[(cast5+7)];
    acc3[0] = (acc3[0]+val6);
    acc3[1] = (acc3[1]+val7);
    acc3[2] = (acc3[2]+val8);
    acc3[3] = (acc3[3]+val9);
    acc3[4] = (acc3[4]+val10);
    acc3[5] = (acc3[5]+val11);
    acc3[6] = (acc3[6]+val12);
    acc3[7] = (acc3[7]+val13);
  }
  var cast6 = bitcast<i32>((cast1<<3u));
  var alu108 = (lidx0==0);
  if (alu108) {
    data0_1200[cast6] = acc3[0];
  }
  if (alu108) {
    data0_1200[(cast6+1)] = acc3[2];
  }
  if (alu108) {
    data0_1200[(cast6+2)] = acc3[4];
  }
  if (alu108) {
    data0_1200[(cast6+3)] = acc3[6];
  }
  if (alu108) {
    data0_1200[(cast6+4)] = acc3[1];
  }
  if (alu108) {
    data0_1200[(cast6+5)] = acc3[3];
  }
  if (alu108) {
    data0_1200[(cast6+6)] = acc3[5];
  }
  if (alu108) {
    data0_1200[(cast6+7)] = acc3[7];
  }
}`;

const r_300_32_32_4_2_2_4_2 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32,256>;
@group(0) @binding(1)var<storage,read_write>data0_76800:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_15600:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_1200:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_131072:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_256:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,8>;
  var acc1: array<f32,8>;
  var gidx1 = i32(gindex.y); /* 300 */
  var cast0 = bitcast<u32>(gidx1);
  var cast1 = bitcast<i32>((cast0<<2u));
  var alu0 = (cast1+1);
  var val0 = data1_15600[alu0];
  var val1 = data2_1200[alu0];
  var alu1 = (cast1+3);
  var val2 = data2_1200[alu1];
  var val3 = data1_15600[cast1];
  var val4 = data2_1200[cast1];
  var alu2 = (cast1+2);
  var val5 = data2_1200[alu2];
  var val6 = data1_15600[alu2];
  var val7 = data1_15600[alu1];
  var gidx0 = i32(gindex.x); /* 32 */
  var lidx0 = i32(lindex.x); /* 32 */
  var cast2 = bitcast<u32>(gidx0);
  var cast3 = bitcast<u32>(lidx0);
  var alu3 = ((val3*val5)+val4);
  var alu4 = ((val0*val2)+val1);
  var alu5 = (exp2((val6*1.4426950408889634f))*val5);
  var alu6 = (exp2((val7*1.4426950408889634f))*val2);
  var alu7 = (lidx0<8);
  var alu8 = (lidx0<24);
  var alu9 = ((7<lidx0)&(lidx0<16));
  var alu10 = ((15<lidx0)&alu8);
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  acc0[4] = 0.0f;
  acc0[5] = 0.0f;
  acc0[6] = 0.0f;
  acc0[7] = 0.0f;
  for (var Ridx0_0 = 0; Ridx0_0 < 2; Ridx0_0++) {
    var alu19 = (bitcast<i32>((cast3<<4u))+bitcast<i32>((bitcast<u32>(Ridx0_0)<<3u)));
    var alu20 = (alu19+bitcast<i32>((cast2<<12u)));
    var val8 = data3_131072[alu20];
    var val9 = data3_131072[(alu20+1)];
    var val10 = data3_131072[(alu20+2)];
    var val11 = data3_131072[(alu20+3)];
    var val12 = data3_131072[(alu20+4)];
    var val13 = data3_131072[(alu20+5)];
    var val14 = data3_131072[(alu20+6)];
    var val15 = data3_131072[(alu20+7)];
    var val16 = data3_131072[(alu20+517)];
    var val17 = data3_131072[(alu20+518)];
    var val18 = data3_131072[(alu20+519)];
    var val19 = data3_131072[(alu20+2055)];
    var val20 = data3_131072[(alu20+512)];
    var val21 = data3_131072[(alu20+513)];
    var val22 = data3_131072[(alu20+514)];
    var val23 = data3_131072[(alu20+515)];
    var val24 = data3_131072[(alu20+516)];
    var val25 = data3_131072[(alu20+2048)];
    var val26 = data3_131072[(alu20+2049)];
    var val27 = data3_131072[(alu20+2050)];
    var val28 = data3_131072[(alu20+2051)];
    var val29 = data3_131072[(alu20+2052)];
    var val30 = data3_131072[(alu20+2053)];
    var val31 = data3_131072[(alu20+2054)];
    var val32 = data3_131072[(alu20+1024)];
    var val33 = data3_131072[(alu20+1025)];
    var val34 = data3_131072[(alu20+1026)];
    var val35 = data3_131072[(alu20+1027)];
    var val36 = data3_131072[(alu20+1028)];
    var val37 = data3_131072[(alu20+1029)];
    var val38 = data3_131072[(alu20+1030)];
    var val39 = data3_131072[(alu20+1031)];
    var val40 = data3_131072[(alu20+2560)];
    var val41 = data3_131072[(alu20+2561)];
    var val42 = data3_131072[(alu20+2562)];
    var val43 = data3_131072[(alu20+2563)];
    var val44 = data3_131072[(alu20+2564)];
    var val45 = data3_131072[(alu20+2565)];
    var val46 = data3_131072[(alu20+2566)];
    var val47 = data3_131072[(alu20+2567)];
    var val48 = data3_131072[(alu20+1541)];
    var val49 = data3_131072[(alu20+1542)];
    var val50 = data3_131072[(alu20+1543)];
    var val51 = data3_131072[(alu20+3072)];
    var val52 = data3_131072[(alu20+3073)];
    var val53 = data3_131072[(alu20+3074)];
    var val54 = data3_131072[(alu20+3075)];
    var val55 = data3_131072[(alu20+3076)];
    var val56 = data3_131072[(alu20+3077)];
    var val57 = data3_131072[(alu20+3078)];
    var val58 = data3_131072[(alu20+3079)];
    var val59 = data3_131072[(alu20+1536)];
    var val60 = data3_131072[(alu20+1537)];
    var val61 = data3_131072[(alu20+1538)];
    var val62 = data3_131072[(alu20+1539)];
    var val63 = data3_131072[(alu20+1540)];
    var val64 = data3_131072[(alu20+3584)];
    var val65 = data3_131072[(alu20+3585)];
    var val66 = data3_131072[(alu20+3586)];
    var val67 = data3_131072[(alu20+3587)];
    var val68 = data3_131072[(alu20+3588)];
    var val69 = data3_131072[(alu20+3589)];
    var val70 = data3_131072[(alu20+3590)];
    var val71 = data3_131072[(alu20+3591)];
    var alu21 = (((f32((alu19+-383)))+-1.0f)*0.5f);
    var alu22 = trunc(alu21);
    var alu23 = select(alu22,(alu22+-1.0f),(alu21<alu22));
    var alu24 = (((f32((alu19+-382)))+-1.0f)*0.5f);
    var alu25 = trunc(alu24);
    var alu26 = select(alu25,(alu25+-1.0f),(alu24<alu25));
    var alu27 = (((f32((alu19+-381)))+-1.0f)*0.5f);
    var alu28 = trunc(alu27);
    var alu29 = select(alu28,(alu28+-1.0f),(alu27<alu28));
    var alu30 = (((f32((alu19+-380)))+-1.0f)*0.5f);
    var alu31 = trunc(alu30);
    var alu32 = select(alu31,(alu31+-1.0f),(alu30<alu31));
    var alu33 = (((f32((alu19+-379)))+-1.0f)*0.5f);
    var alu34 = trunc(alu33);
    var alu35 = select(alu34,(alu34+-1.0f),(alu33<alu34));
    var alu36 = (((f32((alu19+-378)))+-1.0f)*0.5f);
    var alu37 = trunc(alu36);
    var alu38 = select(alu37,(alu37+-1.0f),(alu36<alu37));
    var alu39 = (((f32((alu19+-377)))+-1.0f)*0.5f);
    var alu40 = trunc(alu39);
    var alu41 = select(alu40,(alu40+-1.0f),(alu39<alu40));
    var alu42 = (((f32((alu19+-376)))+-1.0f)*0.5f);
    var alu43 = trunc(alu42);
    var alu44 = select(alu43,(alu43+-1.0f),(alu42<alu43));
    var alu45 = (((f32((alu19+-255)))+-1.0f)*0.5f);
    var alu46 = trunc(alu45);
    var alu47 = select(alu46,(alu46+-1.0f),(alu45<alu46));
    var alu48 = (((f32((alu19+-254)))+-1.0f)*0.5f);
    var alu49 = trunc(alu48);
    var alu50 = select(alu49,(alu49+-1.0f),(alu48<alu49));
    var alu51 = (((f32((alu19+-253)))+-1.0f)*0.5f);
    var alu52 = trunc(alu51);
    var alu53 = select(alu52,(alu52+-1.0f),(alu51<alu52));
    var alu54 = (((f32((alu19+-252)))+-1.0f)*0.5f);
    var alu55 = trunc(alu54);
    var alu56 = select(alu55,(alu55+-1.0f),(alu54<alu55));
    var alu57 = (((f32((alu19+-251)))+-1.0f)*0.5f);
    var alu58 = trunc(alu57);
    var alu59 = select(alu58,(alu58+-1.0f),(alu57<alu58));
    var alu60 = (((f32((alu19+-250)))+-1.0f)*0.5f);
    var alu61 = trunc(alu60);
    var alu62 = select(alu61,(alu61+-1.0f),(alu60<alu61));
    var alu63 = (((f32((alu19+-249)))+-1.0f)*0.5f);
    var alu64 = trunc(alu63);
    var alu65 = select(alu64,(alu64+-1.0f),(alu63<alu64));
    var alu66 = (((f32((alu19+-248)))+-1.0f)*0.5f);
    var alu67 = trunc(alu66);
    var alu68 = select(alu67,(alu67+-1.0f),(alu66<alu67));
    var alu69 = (((f32((alu19+-127)))+-1.0f)*0.5f);
    var alu70 = trunc(alu69);
    var alu71 = select(alu70,(alu70+-1.0f),(alu69<alu70));
    var alu72 = (((f32((alu19+-126)))+-1.0f)*0.5f);
    var alu73 = trunc(alu72);
    var alu74 = select(alu73,(alu73+-1.0f),(alu72<alu73));
    var alu75 = (((f32((alu19+-125)))+-1.0f)*0.5f);
    var alu76 = trunc(alu75);
    var alu77 = select(alu76,(alu76+-1.0f),(alu75<alu76));
    var alu78 = (((f32((alu19+-124)))+-1.0f)*0.5f);
    var alu79 = trunc(alu78);
    var alu80 = select(alu79,(alu79+-1.0f),(alu78<alu79));
    var alu81 = (((f32((alu19+-123)))+-1.0f)*0.5f);
    var alu82 = trunc(alu81);
    var alu83 = select(alu82,(alu82+-1.0f),(alu81<alu82));
    var alu84 = (((f32((alu19+-122)))+-1.0f)*0.5f);
    var alu85 = trunc(alu84);
    var alu86 = select(alu85,(alu85+-1.0f),(alu84<alu85));
    var alu87 = (((f32((alu19+-121)))+-1.0f)*0.5f);
    var alu88 = trunc(alu87);
    var alu89 = select(alu88,(alu88+-1.0f),(alu87<alu88));
    var alu90 = (((f32((alu19+-120)))+-1.0f)*0.5f);
    var alu91 = trunc(alu90);
    var alu92 = select(alu91,(alu91+-1.0f),(alu90<alu91));
    var alu93 = (((f32((alu19+1)))+-1.0f)*0.5f);
    var alu94 = trunc(alu93);
    var alu95 = select(alu94,(alu94+-1.0f),(alu93<alu94));
    var alu96 = (((f32((alu19+2)))+-1.0f)*0.5f);
    var alu97 = trunc(alu96);
    var alu98 = select(alu97,(alu97+-1.0f),(alu96<alu97));
    var alu99 = (((f32((alu19+3)))+-1.0f)*0.5f);
    var alu100 = trunc(alu99);
    var alu101 = select(alu100,(alu100+-1.0f),(alu99<alu100));
    var alu102 = (((f32((alu19+4)))+-1.0f)*0.5f);
    var alu103 = trunc(alu102);
    var alu104 = select(alu103,(alu103+-1.0f),(alu102<alu103));
    var alu105 = (((f32((alu19+5)))+-1.0f)*0.5f);
    var alu106 = trunc(alu105);
    var alu107 = select(alu106,(alu106+-1.0f),(alu105<alu106));
    var alu108 = (((f32((alu19+6)))+-1.0f)*0.5f);
    var alu109 = trunc(alu108);
    var alu110 = select(alu109,(alu109+-1.0f),(alu108<alu109));
    var alu111 = (((f32((alu19+7)))+-1.0f)*0.5f);
    var alu112 = trunc(alu111);
    var alu113 = select(alu112,(alu112+-1.0f),(alu111<alu112));
    var alu114 = (((f32((alu19+8)))+-1.0f)*0.5f);
    var alu115 = trunc(alu114);
    var alu116 = select(alu115,(alu115+-1.0f),(alu114<alu115));
    var alu117 = select(0.0f,(alu4*(1/exp2((alu98*0.20762050593046014f)))*6.283185307179586f),alu7);
    var alu118 = select(0.0f,(alu4*(1/exp2((alu104*0.20762050593046014f)))*6.283185307179586f),alu7);
    var alu119 = select(0.0f,(alu4*(1/exp2((alu110*0.20762050593046014f)))*6.283185307179586f),alu7);
    var alu120 = select(0.0f,(alu4*(1/exp2((alu116*0.20762050593046014f)))*6.283185307179586f),alu7);
    var alu121 = select((alu6*(1/exp2((alu26*0.20762050593046014f)))*6.283185307179586f),0.0f,alu8);
    var alu122 = select((alu6*(1/exp2((alu32*0.20762050593046014f)))*6.283185307179586f),0.0f,alu8);
    var alu123 = select((alu6*(1/exp2((alu38*0.20762050593046014f)))*6.283185307179586f),0.0f,alu8);
    var alu124 = select((alu6*(1/exp2((alu44*0.20762050593046014f)))*6.283185307179586f),0.0f,alu8);
    var alu125 = select(0.0f,(alu3*(1/exp2((alu74*0.20762050593046014f)))*6.283185307179586f),alu9);
    var alu126 = select(0.0f,(alu3*(1/exp2((alu80*0.20762050593046014f)))*6.283185307179586f),alu9);
    var alu127 = select(0.0f,(alu3*(1/exp2((alu86*0.20762050593046014f)))*6.283185307179586f),alu9);
    var alu128 = select(0.0f,(alu3*(1/exp2((alu92*0.20762050593046014f)))*6.283185307179586f),alu9);
    var alu129 = select(0.0f,(alu5*(1/exp2((alu50*0.20762050593046014f)))*6.283185307179586f),alu10);
    var alu130 = select(0.0f,(alu5*(1/exp2((alu56*0.20762050593046014f)))*6.283185307179586f),alu10);
    var alu131 = select(0.0f,(alu5*(1/exp2((alu62*0.20762050593046014f)))*6.283185307179586f),alu10);
    var alu132 = select(0.0f,(alu5*(1/exp2((alu68*0.20762050593046014f)))*6.283185307179586f),alu10);
    var alu133 = select(0.0f,sin((alu4*(1/exp2((alu95*0.20762050593046014f)))*6.283185307179586f)),alu7);
    var alu134 = select(0.0f,sin((alu3*(1/exp2((alu71*0.20762050593046014f)))*6.283185307179586f)),alu9);
    var alu135 = select(0.0f,sin((alu5*(1/exp2((alu47*0.20762050593046014f)))*6.283185307179586f)),alu10);
    var alu136 = select(sin((alu6*(1/exp2((alu23*0.20762050593046014f)))*6.283185307179586f)),0.0f,alu8);
    var alu137 = select(alu136,0.0f,alu8);
    var alu138 = (alu133+alu134+alu135+alu137);
    var alu139 = select(0.0f,sin((1.5707963267948966f-alu117)),alu7);
    var alu140 = select(0.0f,sin((1.5707963267948966f-alu125)),alu9);
    var alu141 = select(0.0f,sin((1.5707963267948966f-alu129)),alu10);
    var alu142 = select(sin((1.5707963267948966f-alu121)),0.0f,alu8);
    var alu143 = select(alu142,0.0f,alu8);
    var alu144 = (alu139+alu140+alu141+alu143);
    var alu145 = select(0.0f,sin((alu4*(1/exp2((alu101*0.20762050593046014f)))*6.283185307179586f)),alu7);
    var alu146 = select(0.0f,sin((alu3*(1/exp2((alu77*0.20762050593046014f)))*6.283185307179586f)),alu9);
    var alu147 = select(0.0f,sin((alu5*(1/exp2((alu53*0.20762050593046014f)))*6.283185307179586f)),alu10);
    var alu148 = select(sin((alu6*(1/exp2((alu29*0.20762050593046014f)))*6.283185307179586f)),0.0f,alu8);
    var alu149 = select(alu148,0.0f,alu8);
    var alu150 = (alu145+alu146+alu147+alu149);
    var alu151 = select(0.0f,sin((1.5707963267948966f-alu118)),alu7);
    var alu152 = select(0.0f,sin((1.5707963267948966f-alu126)),alu9);
    var alu153 = select(0.0f,sin((1.5707963267948966f-alu130)),alu10);
    var alu154 = select(sin((1.5707963267948966f-alu122)),0.0f,alu8);
    var alu155 = select(alu154,0.0f,alu8);
    var alu156 = (alu151+alu152+alu153+alu155);
    var alu157 = select(0.0f,sin((alu4*(1/exp2((alu107*0.20762050593046014f)))*6.283185307179586f)),alu7);
    var alu158 = select(0.0f,sin((alu3*(1/exp2((alu83*0.20762050593046014f)))*6.283185307179586f)),alu9);
    var alu159 = select(0.0f,sin((alu5*(1/exp2((alu59*0.20762050593046014f)))*6.283185307179586f)),alu10);
    var alu160 = select(sin((alu6*(1/exp2((alu35*0.20762050593046014f)))*6.283185307179586f)),0.0f,alu8);
    var alu161 = select(alu160,0.0f,alu8);
    var alu162 = (alu157+alu158+alu159+alu161);
    var alu163 = select(0.0f,sin((1.5707963267948966f-alu119)),alu7);
    var alu164 = select(0.0f,sin((1.5707963267948966f-alu127)),alu9);
    var alu165 = select(0.0f,sin((1.5707963267948966f-alu131)),alu10);
    var alu166 = select(sin((1.5707963267948966f-alu123)),0.0f,alu8);
    var alu167 = select(alu166,0.0f,alu8);
    var alu168 = (alu163+alu164+alu165+alu167);
    var alu169 = select(0.0f,sin((alu4*(1/exp2((alu113*0.20762050593046014f)))*6.283185307179586f)),alu7);
    var alu170 = select(0.0f,sin((alu3*(1/exp2((alu89*0.20762050593046014f)))*6.283185307179586f)),alu9);
    var alu171 = select(0.0f,sin((alu5*(1/exp2((alu65*0.20762050593046014f)))*6.283185307179586f)),alu10);
    var alu172 = select(sin((alu6*(1/exp2((alu41*0.20762050593046014f)))*6.283185307179586f)),0.0f,alu8);
    var alu173 = select(alu172,0.0f,alu8);
    var alu174 = (alu169+alu170+alu171+alu173);
    var alu175 = select(0.0f,sin((1.5707963267948966f-alu120)),alu7);
    var alu176 = select(0.0f,sin((1.5707963267948966f-alu128)),alu9);
    var alu177 = select(0.0f,sin((1.5707963267948966f-alu132)),alu10);
    var alu178 = select(sin((1.5707963267948966f-alu124)),0.0f,alu8);
    var alu179 = select(alu178,0.0f,alu8);
    var alu180 = (alu175+alu176+alu177+alu179);
    acc0[0] = (acc0[0]+(alu138*val8)+(alu144*val9)+(alu150*val10)+(alu156*val11)+(alu162*val12)+(alu168*val13)+(alu174*val14)+(alu180*val15));
    acc0[1] = (acc0[1]+(alu138*val25)+(alu144*val26)+(alu150*val27)+(alu156*val28)+(alu162*val29)+(alu168*val30)+(alu174*val31)+(alu180*val19));
    acc0[2] = (acc0[2]+(alu138*val20)+(alu144*val21)+(alu150*val22)+(alu156*val23)+(alu162*val24)+(alu168*val16)+(alu174*val17)+(alu180*val18));
    acc0[3] = (acc0[3]+(alu138*val40)+(alu144*val41)+(alu150*val42)+(alu156*val43)+(alu162*val44)+(alu168*val45)+(alu174*val46)+(alu180*val47));
    acc0[4] = (acc0[4]+(alu138*val32)+(alu144*val33)+(alu150*val34)+(alu156*val35)+(alu162*val36)+(alu168*val37)+(alu174*val38)+(alu180*val39));
    acc0[5] = (acc0[5]+(alu138*val51)+(alu144*val52)+(alu150*val53)+(alu156*val54)+(alu162*val55)+(alu168*val56)+(alu174*val57)+(alu180*val58));
    acc0[6] = (acc0[6]+(alu138*val59)+(alu144*val60)+(alu150*val61)+(alu156*val62)+(alu162*val63)+(alu168*val48)+(alu174*val49)+(alu180*val50));
    acc0[7] = (acc0[7]+(alu138*val64)+(alu144*val65)+(alu150*val66)+(alu156*val67)+(alu162*val68)+(alu168*val69)+(alu174*val70)+(alu180*val71));
  }
  var cast4 = bitcast<i32>((cast3<<3u));
  temp0[cast4] = acc0[0];
  temp0[(cast4+1)] = acc0[1];
  temp0[(cast4+2)] = acc0[2];
  temp0[(cast4+3)] = acc0[3];
  temp0[(cast4+4)] = acc0[4];
  temp0[(cast4+5)] = acc0[5];
  temp0[(cast4+6)] = acc0[6];
  temp0[(cast4+7)] = acc0[7];
  workgroupBarrier();
  acc1[0] = 0.0f;
  acc1[1] = 0.0f;
  acc1[2] = 0.0f;
  acc1[3] = 0.0f;
  acc1[4] = 0.0f;
  acc1[5] = 0.0f;
  acc1[6] = 0.0f;
  acc1[7] = 0.0f;
  for (var Ridx103 = 0; Ridx103 < 32; Ridx103++) {
    var cast5 = bitcast<i32>((bitcast<u32>(Ridx103)<<3u));
    var val72 = temp0[cast5];
    var val73 = temp0[(cast5+1)];
    var val74 = temp0[(cast5+2)];
    var val75 = temp0[(cast5+3)];
    var val76 = temp0[(cast5+4)];
    var val77 = temp0[(cast5+5)];
    var val78 = temp0[(cast5+6)];
    var val79 = temp0[(cast5+7)];
    acc1[0] = (acc1[0]+val72);
    acc1[1] = (acc1[1]+val73);
    acc1[2] = (acc1[2]+val74);
    acc1[3] = (acc1[3]+val75);
    acc1[4] = (acc1[4]+val76);
    acc1[5] = (acc1[5]+val77);
    acc1[6] = (acc1[6]+val78);
    acc1[7] = (acc1[7]+val79);
  }
  var cast6 = bitcast<i32>((cast2<<3u));
  var val80 = data4_256[cast6];
  var val81 = data4_256[(cast6+1)];
  var val82 = data4_256[(cast6+2)];
  var val83 = data4_256[(cast6+3)];
  var val84 = data4_256[(cast6+4)];
  var val85 = data4_256[(cast6+5)];
  var val86 = data4_256[(cast6+6)];
  var val87 = data4_256[(cast6+7)];
  var alu216 = (cast6+bitcast<i32>((cast0<<8u)));
  var alu217 = (lidx0==0);
  var alu218 = (acc1[0]+val80);
  var alu219 = (acc1[1]+val84);
  var alu220 = (acc1[2]+val81);
  var alu221 = (acc1[3]+val85);
  var alu222 = (acc1[4]+val82);
  var alu223 = (acc1[5]+val86);
  var alu224 = (acc1[6]+val83);
  var alu225 = (acc1[7]+val87);
  var alu226 = select(0.0f,alu218,(0.0f<alu218));
  var alu227 = select(0.0f,alu219,(0.0f<alu219));
  var alu228 = select(0.0f,alu220,(0.0f<alu220));
  var alu229 = select(0.0f,alu221,(0.0f<alu221));
  var alu230 = select(0.0f,alu222,(0.0f<alu222));
  var alu231 = select(0.0f,alu223,(0.0f<alu223));
  var alu232 = select(0.0f,alu224,(0.0f<alu224));
  var alu233 = select(0.0f,alu225,(0.0f<alu225));
  if (alu217) {
    data0_76800[alu216] = alu226;
  }
  if (alu217) {
    data0_76800[(alu216+1)] = alu228;
  }
  if (alu217) {
    data0_76800[(alu216+2)] = alu230;
  }
  if (alu217) {
    data0_76800[(alu216+3)] = alu232;
  }
  if (alu217) {
    data0_76800[(alu216+4)] = alu227;
  }
  if (alu217) {
    data0_76800[(alu216+5)] = alu229;
  }
  if (alu217) {
    data0_76800[(alu216+6)] = alu231;
  }
  if (alu217) {
    data0_76800[(alu216+7)] = alu233;
  }
}`;

const r_15_4_8_4_8_5_256 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_76800:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_76800:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_65536:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_256:array<f32>;
@compute @workgroup_size(8,4,8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,5>;
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 15 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 4 */
  var lidx2 = i32(lindex.z); /* 8 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(lidx2);
  var alu0 = ((gidx1*5120)+bitcast<i32>((bitcast<u32>(lidx1)<<8u)));
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  acc0[4] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 256; Ridx0++) {
    var alu6 = (alu0+Ridx0);
    var val0 = data1_76800[alu6];
    var val1 = data2_65536[(bitcast<i32>((cast0<<14u))+bitcast<i32>((cast1<<11u))+bitcast<i32>((bitcast<u32>(lidx0)<<8u))+Ridx0)];
    var val2 = data1_76800[(alu6+1024)];
    var val3 = data1_76800[(alu6+2048)];
    var val4 = data1_76800[(alu6+3072)];
    var val5 = data1_76800[(alu6+4096)];
    acc0[0] = (acc0[0]+(val0*val1));
    acc0[1] = (acc0[1]+(val2*val1));
    acc0[2] = (acc0[2]+(val3*val1));
    acc0[3] = (acc0[3]+(val4*val1));
    acc0[4] = (acc0[4]+(val5*val1));
  }
  var alu13 = (lidx0+bitcast<i32>((cast0<<6u))+bitcast<i32>((cast1<<3u)));
  var val6 = data3_256[alu13];
  var alu14 = (alu13+alu0);
  data0_76800[alu14] = (acc0[0]+val6);
  data0_76800[(alu14+1024)] = (acc0[1]+val6);
  data0_76800[(alu14+2048)] = (acc0[2]+val6);
  data0_76800[(alu14+3072)] = (acc0[3]+val6);
  data0_76800[(alu14+4096)] = (acc0[4]+val6);
}`;

const r_75_2_16_8_4_256 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_76800:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_998400:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_76800:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_196608:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_768:array<f32>;
@compute @workgroup_size(16,8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 2 */
  var gidx1 = i32(gindex.y); /* 75 */
  var lidx0 = i32(lindex.x); /* 16 */
  var lidx1 = i32(lindex.y); /* 8 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<i32>((bitcast<u32>(gidx1)<<10u));
  var cast2 = bitcast<u32>(lidx1);
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 256; Ridx0++) {
    var alu4 = (cast1+Ridx0);
    var val0 = data1_998400[alu4];
    var val1 = data2_76800[alu4];
    var val2 = data3_196608[(bitcast<i32>((cast0<<15u))+bitcast<i32>((cast2<<12u))+bitcast<i32>((bitcast<u32>(lidx0)<<8u))+Ridx0)];
    var alu5 = (alu4+256);
    var val3 = data1_998400[alu5];
    var val4 = data2_76800[alu5];
    var alu6 = (alu4+512);
    var val5 = data1_998400[alu6];
    var val6 = data2_76800[alu6];
    var alu7 = (alu4+768);
    var val7 = data1_998400[alu7];
    var val8 = data2_76800[alu7];
    acc0[0] = (acc0[0]+((val0+val1)*val2));
    acc0[1] = (acc0[1]+((val3+val4)*val2));
    acc0[2] = (acc0[2]+((val5+val6)*val2));
    acc0[3] = (acc0[3]+((val7+val8)*val2));
  }
  var alu13 = (lidx0+bitcast<i32>((cast0<<7u))+bitcast<i32>((cast2<<4u)));
  var val9 = data4_768[alu13];
  var alu14 = (alu13+cast1);
  data0_76800[alu14] = (acc0[0]+val9);
  data0_76800[(alu14+256)] = (acc0[1]+val9);
  data0_76800[(alu14+512)] = (acc0[2]+val9);
  data0_76800[(alu14+768)] = (acc0[3]+val9);
}`;

const r_75_32_8_4_256 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_76800:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_998400:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_76800:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_196608:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_768:array<f32>;
@compute @workgroup_size(8,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 32 */
  var gidx1 = i32(gindex.y); /* 75 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 4 */
  var cast0 = bitcast<u32>(gidx0);
  var alu0 = (bitcast<i32>((bitcast<u32>(gidx1)<<10u))+bitcast<i32>((bitcast<u32>(lidx1)<<8u)));
  acc0[0] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 256; Ridx0++) {
    var alu2 = (alu0+Ridx0);
    var val0 = data1_998400[alu2];
    var val1 = data2_76800[alu2];
    var val2 = data3_196608[(bitcast<i32>((cast0<<11u))+bitcast<i32>((bitcast<u32>(lidx0)<<8u))+Ridx0+65536)];
    acc0[0] = (acc0[0]+((val0+val1)*val2));
  }
  var alu5 = (lidx0+bitcast<i32>((cast0<<3u)));
  var val3 = data4_768[(alu5+256)];
  data0_76800[(alu5+alu0)] = (acc0[0]+val3);
}`;

const r_2_300_75_4_4_32 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_720000:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_76800:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_76800:array<f32>;
@compute @workgroup_size(4,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 75 */
  var gidx1 = i32(gindex.y); /* 300 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 4 */
  var lidx1 = i32(lindex.y); /* 4 */
  var cast0 = bitcast<u32>(gidx0);
  acc0[0] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 32; Ridx0++) {
    var alu1 = (bitcast<i32>((bitcast<u32>(gidx2)<<7u))+bitcast<i32>((bitcast<u32>(lidx0)<<5u))+Ridx0);
    var val0 = data1_76800[(alu1+bitcast<i32>((bitcast<u32>(gidx1)<<8u)))];
    var val1 = data2_76800[(alu1+bitcast<i32>((cast0<<10u))+bitcast<i32>((bitcast<u32>(lidx1)<<8u)))];
    acc0[0] = (acc0[0]+(val0*val1));
  }
  data0_720000[(lidx1+bitcast<i32>((cast0<<2u))+(gidx1*300)+(gidx2*360000)+(lidx0*90000))] = (acc0[0]*0.17677669529663687f);
}`;

const r_600_4_75_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_2400:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_720000:array<f32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 600 */
  acc0[0] = (f32(-INFINITY));
  acc0[1] = (f32(-INFINITY));
  acc0[2] = (f32(-INFINITY));
  acc0[3] = (f32(-INFINITY));
  for (var Ridx0 = 0; Ridx0 < 75; Ridx0++) {
    var alu4 = ((gidx0*1200)+bitcast<i32>((bitcast<u32>(Ridx0)<<2u)));
    var val0 = data1_720000[(alu4+1)];
    var val1 = data1_720000[(alu4+2)];
    var val2 = data1_720000[(alu4+3)];
    var val3 = data1_720000[(alu4+300)];
    var val4 = data1_720000[(alu4+301)];
    var val5 = data1_720000[(alu4+302)];
    var val6 = data1_720000[(alu4+303)];
    var val7 = data1_720000[alu4];
    var val8 = data1_720000[(alu4+600)];
    var val9 = data1_720000[(alu4+601)];
    var val10 = data1_720000[(alu4+602)];
    var val11 = data1_720000[(alu4+603)];
    var val12 = data1_720000[(alu4+900)];
    var val13 = data1_720000[(alu4+901)];
    var val14 = data1_720000[(alu4+902)];
    var val15 = data1_720000[(alu4+903)];
    var alu5 = select(acc0[0],val7,(acc0[0]<val7));
    var alu6 = select(acc0[1],val3,(acc0[1]<val3));
    var alu7 = select(acc0[2],val8,(acc0[2]<val8));
    var alu8 = select(acc0[3],val12,(acc0[3]<val12));
    var alu9 = select(alu5,val0,(alu5<val0));
    var alu10 = select(alu6,val4,(alu6<val4));
    var alu11 = select(alu7,val9,(alu7<val9));
    var alu12 = select(alu8,val13,(alu8<val13));
    var alu13 = select(alu9,val1,(alu9<val1));
    var alu14 = select(alu10,val5,(alu10<val5));
    var alu15 = select(alu11,val10,(alu11<val10));
    var alu16 = select(alu12,val14,(alu12<val14));
    var alu17 = select(alu13,val2,(alu13<val2));
    var alu18 = select(alu14,val6,(alu14<val6));
    var alu19 = select(alu15,val11,(alu15<val11));
    var alu20 = select(alu16,val15,(alu16<val15));
    acc0[0] = alu17;
    acc0[1] = alu18;
    acc0[2] = alu19;
    acc0[3] = alu20;
  }
  var cast0 = bitcast<i32>((bitcast<u32>(gidx0)<<2u));
  data0_2400[cast0] = acc0[0];
  data0_2400[(cast0+1)] = acc0[1];
  data0_2400[(cast0+2)] = acc0[2];
  data0_2400[(cast0+3)] = acc0[3];
}`;

const r_150_16_75_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_2400:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_720000:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_2400:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 150 */
  var lidx0 = i32(lindex.x); /* 16 */
  var alu0 = (lidx0+bitcast<i32>((bitcast<u32>(gidx0)<<4u)));
  var val0 = data2_2400[alu0];
  acc0[0] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 75; Ridx0++) {
    var alu2 = ((gidx0*4800)+(lidx0*300)+bitcast<i32>((bitcast<u32>(Ridx0)<<2u)));
    var val1 = data1_720000[(alu2+1)];
    var val2 = data1_720000[(alu2+2)];
    var val3 = data1_720000[(alu2+3)];
    var val4 = data1_720000[alu2];
    acc0[0] = (acc0[0]+exp2(((val4-val0)*1.4426950408889634f))+exp2(((val1-val0)*1.4426950408889634f))+exp2(((val2-val0)*1.4426950408889634f))+exp2(((val3-val0)*1.4426950408889634f)));
  }
  data0_2400[alu0] = acc0[0];
}`;

const r_60_8_16_5_2_300 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_76800:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_720000:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_2400:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_2400:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_76800:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,10>;
  var gidx0 = i32(gindex.x); /* 8 */
  var gidx1 = i32(gindex.y); /* 60 */
  var alu0 = ((gidx0*300)+(gidx1*5));
  var alu1 = (alu0+1);
  var val0 = data2_2400[alu1];
  var alu2 = (alu0+2);
  var val1 = data2_2400[alu2];
  var alu3 = (alu0+3);
  var val2 = data2_2400[alu3];
  var alu4 = (alu0+4);
  var val3 = data2_2400[alu4];
  var val4 = data2_2400[alu0];
  var lidx0 = i32(lindex.x); /* 16 */
  var alu5 = (lidx0+bitcast<i32>((bitcast<u32>(gidx0)<<5u)));
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  acc0[4] = 0.0f;
  acc0[5] = 0.0f;
  acc0[6] = 0.0f;
  acc0[7] = 0.0f;
  acc0[8] = 0.0f;
  acc0[9] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 300; Ridx0++) {
    var alu16 = ((gidx1*1500)+Ridx0+(gidx0*90000));
    var val5 = data1_720000[alu16];
    var alu17 = (alu5+bitcast<i32>((bitcast<u32>(Ridx0)<<8u)));
    var val6 = data4_76800[alu17];
    var val7 = data4_76800[(alu17+16)];
    var val8 = data1_720000[(alu16+300)];
    var val9 = data1_720000[(alu16+600)];
    var val10 = data1_720000[(alu16+900)];
    var val11 = data1_720000[(alu16+1200)];
    var alu18 = exp2(((val8-val0)*1.4426950408889634f));
    var alu19 = exp2(((val9-val1)*1.4426950408889634f));
    var alu20 = exp2(((val10-val2)*1.4426950408889634f));
    var alu21 = exp2(((val11-val3)*1.4426950408889634f));
    var alu22 = exp2(((val5-val4)*1.4426950408889634f));
    acc0[0] = (acc0[0]+(alu22*val6));
    acc0[1] = (acc0[1]+(alu22*val7));
    acc0[2] = (acc0[2]+(alu18*val6));
    acc0[3] = (acc0[3]+(alu18*val7));
    acc0[4] = (acc0[4]+(alu19*val6));
    acc0[5] = (acc0[5]+(alu19*val7));
    acc0[6] = (acc0[6]+(alu20*val6));
    acc0[7] = (acc0[7]+(alu20*val7));
    acc0[8] = (acc0[8]+(alu21*val6));
    acc0[9] = (acc0[9]+(alu21*val7));
  }
  var val12 = data3_2400[alu0];
  var val13 = data3_2400[alu1];
  var val14 = data3_2400[alu2];
  var val15 = data3_2400[alu3];
  var val16 = data3_2400[alu4];
  var alu34 = (alu5+(gidx1*1280));
  var alu35 = (1/val13);
  var alu36 = (1/val14);
  var alu37 = (1/val15);
  var alu38 = (1/val16);
  var alu39 = (1/val12);
  data0_76800[alu34] = (acc0[0]*alu39);
  data0_76800[(alu34+16)] = (acc0[1]*alu39);
  data0_76800[(alu34+256)] = (acc0[2]*alu35);
  data0_76800[(alu34+272)] = (acc0[3]*alu35);
  data0_76800[(alu34+512)] = (acc0[4]*alu36);
  data0_76800[(alu34+528)] = (acc0[5]*alu36);
  data0_76800[(alu34+768)] = (acc0[6]*alu37);
  data0_76800[(alu34+784)] = (acc0[7]*alu37);
  data0_76800[(alu34+1024)] = (acc0[8]*alu38);
  data0_76800[(alu34+1040)] = (acc0[9]*alu38);
}`;

const r_300_16_16_256 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_76800:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_998400:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_76800:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_65536:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_256:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 300 */
  var lidx0 = i32(lindex.x); /* 16 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<i32>((bitcast<u32>(gidx1)<<8u));
  acc0[0] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 256; Ridx0++) {
    var val0 = data2_76800[(cast1+Ridx0)];
    var val1 = data3_65536[(bitcast<i32>((cast0<<12u))+bitcast<i32>((bitcast<u32>(lidx0)<<8u))+Ridx0)];
    acc0[0] = (acc0[0]+(val0*val1));
  }
  var alu3 = (lidx0+bitcast<i32>((cast0<<4u)));
  var alu4 = (alu3+cast1);
  var val2 = data1_998400[alu4];
  var val3 = data4_256[alu3];
  data0_76800[alu4] = (val2+acc0[0]+val3);
}`;

const r_150_2_256 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_300:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_76800:array<f32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,2>;
  var gidx0 = i32(gindex.x); /* 150 */
  var cast0 = bitcast<u32>(gidx0);
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 256; Ridx0++) {
    var alu2 = (bitcast<i32>((cast0<<9u))+Ridx0);
    var val0 = data1_76800[alu2];
    var val1 = data1_76800[(alu2+256)];
    acc0[0] = (acc0[0]+val0);
    acc0[1] = (acc0[1]+val1);
  }
  var cast1 = bitcast<i32>((cast0<<1u));
  data0_300[cast1] = (acc0[0]*0.00390625f);
  data0_300[(cast1+1)] = (acc0[1]*0.00390625f);
}`;

const r_300_16_16 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32,16>;
@group(0) @binding(1)var<storage,read_write>data0_300:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_76800:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_300:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,1>;
  var acc1: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 300 */
  var val0 = data2_300[gidx0];
  var lidx0 = i32(lindex.x); /* 16 */
  acc0[0] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 16; Ridx0++) {
    var val1 = data1_76800[(bitcast<i32>((bitcast<u32>(lidx0)<<4u))+Ridx0+bitcast<i32>((bitcast<u32>(gidx0)<<8u)))];
    var alu1 = (val1-val0);
    acc0[0] = (acc0[0]+(alu1*alu1));
  }
  temp0[lidx0] = acc0[0];
  workgroupBarrier();
  acc1[0] = 0.0f;
  for (var Ridx102 = 0; Ridx102 < 16; Ridx102++) {
    var val2 = temp0[Ridx102];
    acc1[0] = (acc1[0]+val2);
  }
  var alu9 = (lidx0==0);
  if (alu9) {
    data0_300[gidx0] = (1/sqrt(((acc1[0]*0.00390625f)+1e-05f)));
  }
}`;

const E_300_16_16 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_76800:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_76800:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_300:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_300:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_256:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_256:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 300 */
  var lidx0 = i32(lindex.x); /* 16 */
  var alu0 = (lidx0+bitcast<i32>((bitcast<u32>(gidx0)<<4u)));
  var alu1 = (alu0+bitcast<i32>((bitcast<u32>(gidx1)<<8u)));
  var val0 = data1_76800[alu1];
  var val1 = data2_300[gidx1];
  var val2 = data3_300[gidx1];
  var val3 = data4_256[alu0];
  var val4 = data5_256[alu0];
  data0_76800[alu1] = (((val0-val1)*val2*val3)+val4);
}`;

const r_150_2_2_16_2_16_16 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32,512>;
@group(0) @binding(1)var<storage,read_write>data0_19200:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_15600:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_1200:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_76800:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_76800:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_16384:array<f32>;
@group(0) @binding(7)var<storage,read_write>data6_64:array<f32>;
@compute @workgroup_size(16,2) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,16>;
  var acc1: array<f32,16>;
  var gidx0 = i32(gindex.x); /* 2 */
  var gidx1 = i32(gindex.y); /* 2 */
  var gidx2 = i32(gindex.z); /* 150 */
  var lidx0 = i32(lindex.x); /* 16 */
  var lidx1 = i32(lindex.y); /* 2 */
  var cast0 = bitcast<u32>(gidx1);
  var cast1 = bitcast<u32>(gidx2);
  var cast2 = bitcast<u32>(lidx1);
  var cast3 = bitcast<i32>((cast2<<8u));
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  acc0[4] = 0.0f;
  acc0[5] = 0.0f;
  acc0[6] = 0.0f;
  acc0[7] = 0.0f;
  acc0[8] = 0.0f;
  acc0[9] = 0.0f;
  acc0[10] = 0.0f;
  acc0[11] = 0.0f;
  acc0[12] = 0.0f;
  acc0[13] = 0.0f;
  acc0[14] = 0.0f;
  acc0[15] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 16; Ridx0++) {
    var alu16 = (lidx0+bitcast<i32>((bitcast<u32>(Ridx0)<<4u)));
    var alu17 = (alu16+bitcast<i32>((cast1<<9u))+cast3);
    var val0 = data3_76800[alu17];
    var val1 = data4_76800[alu17];
    var alu18 = (alu16+bitcast<i32>((bitcast<u32>(gidx0)<<8u))+bitcast<i32>((cast0<<9u)));
    var val2 = data5_16384[alu18];
    var val3 = data5_16384[(alu18+1024)];
    var val4 = data5_16384[(alu18+2048)];
    var val5 = data5_16384[(alu18+3072)];
    var val6 = data5_16384[(alu18+4096)];
    var val7 = data5_16384[(alu18+5120)];
    var val8 = data5_16384[(alu18+6144)];
    var val9 = data5_16384[(alu18+7168)];
    var val10 = data5_16384[(alu18+8192)];
    var val11 = data5_16384[(alu18+9216)];
    var val12 = data5_16384[(alu18+10240)];
    var val13 = data5_16384[(alu18+11264)];
    var val14 = data5_16384[(alu18+12288)];
    var val15 = data5_16384[(alu18+13312)];
    var val16 = data5_16384[(alu18+14336)];
    var val17 = data5_16384[(alu18+15360)];
    var alu19 = (val0+val1);
    acc0[0] = (acc0[0]+(alu19*val2));
    acc0[1] = (acc0[1]+(alu19*val3));
    acc0[2] = (acc0[2]+(alu19*val4));
    acc0[3] = (acc0[3]+(alu19*val5));
    acc0[4] = (acc0[4]+(alu19*val6));
    acc0[5] = (acc0[5]+(alu19*val7));
    acc0[6] = (acc0[6]+(alu19*val8));
    acc0[7] = (acc0[7]+(alu19*val9));
    acc0[8] = (acc0[8]+(alu19*val10));
    acc0[9] = (acc0[9]+(alu19*val11));
    acc0[10] = (acc0[10]+(alu19*val12));
    acc0[11] = (acc0[11]+(alu19*val13));
    acc0[12] = (acc0[12]+(alu19*val14));
    acc0[13] = (acc0[13]+(alu19*val15));
    acc0[14] = (acc0[14]+(alu19*val16));
    acc0[15] = (acc0[15]+(alu19*val17));
  }
  var alu37 = (bitcast<i32>((bitcast<u32>(lidx0)<<4u))+cast3);
  temp0[alu37] = acc0[0];
  temp0[(alu37+1)] = acc0[1];
  temp0[(alu37+2)] = acc0[2];
  temp0[(alu37+3)] = acc0[3];
  temp0[(alu37+4)] = acc0[4];
  temp0[(alu37+5)] = acc0[5];
  temp0[(alu37+6)] = acc0[6];
  temp0[(alu37+7)] = acc0[7];
  temp0[(alu37+8)] = acc0[8];
  temp0[(alu37+9)] = acc0[9];
  temp0[(alu37+10)] = acc0[10];
  temp0[(alu37+11)] = acc0[11];
  temp0[(alu37+12)] = acc0[12];
  temp0[(alu37+13)] = acc0[13];
  temp0[(alu37+14)] = acc0[14];
  temp0[(alu37+15)] = acc0[15];
  workgroupBarrier();
  acc1[0] = 0.0f;
  acc1[1] = 0.0f;
  acc1[2] = 0.0f;
  acc1[3] = 0.0f;
  acc1[4] = 0.0f;
  acc1[5] = 0.0f;
  acc1[6] = 0.0f;
  acc1[7] = 0.0f;
  acc1[8] = 0.0f;
  acc1[9] = 0.0f;
  acc1[10] = 0.0f;
  acc1[11] = 0.0f;
  acc1[12] = 0.0f;
  acc1[13] = 0.0f;
  acc1[14] = 0.0f;
  acc1[15] = 0.0f;
  for (var Ridx105 = 0; Ridx105 < 16; Ridx105++) {
    var alu71 = (cast3+bitcast<i32>((bitcast<u32>(Ridx105)<<4u)));
    var val18 = temp0[alu71];
    var val19 = temp0[(alu71+1)];
    var val20 = temp0[(alu71+2)];
    var val21 = temp0[(alu71+3)];
    var val22 = temp0[(alu71+4)];
    var val23 = temp0[(alu71+5)];
    var val24 = temp0[(alu71+6)];
    var val25 = temp0[(alu71+7)];
    var val26 = temp0[(alu71+8)];
    var val27 = temp0[(alu71+9)];
    var val28 = temp0[(alu71+10)];
    var val29 = temp0[(alu71+11)];
    var val30 = temp0[(alu71+12)];
    var val31 = temp0[(alu71+13)];
    var val32 = temp0[(alu71+14)];
    var val33 = temp0[(alu71+15)];
    acc1[0] = (acc1[0]+val18);
    acc1[1] = (acc1[1]+val19);
    acc1[2] = (acc1[2]+val20);
    acc1[3] = (acc1[3]+val21);
    acc1[4] = (acc1[4]+val22);
    acc1[5] = (acc1[5]+val23);
    acc1[6] = (acc1[6]+val24);
    acc1[7] = (acc1[7]+val25);
    acc1[8] = (acc1[8]+val26);
    acc1[9] = (acc1[9]+val27);
    acc1[10] = (acc1[10]+val28);
    acc1[11] = (acc1[11]+val29);
    acc1[12] = (acc1[12]+val30);
    acc1[13] = (acc1[13]+val31);
    acc1[14] = (acc1[14]+val32);
    acc1[15] = (acc1[15]+val33);
  }
  var alu89 = (bitcast<i32>((cast1<<3u))+bitcast<i32>((cast2<<2u)));
  var alu90 = (gidx0+alu89);
  var val34 = data1_15600[alu90];
  var val35 = data2_1200[alu90];
  var alu91 = (alu90+2);
  var val36 = data2_1200[alu91];
  var alu92 = (gidx0+bitcast<i32>((cast0<<1u)));
  var val37 = data6_64[alu92];
  var val38 = data1_15600[alu91];
  var val39 = data6_64[(alu92+4)];
  var val40 = data6_64[(alu92+8)];
  var val41 = data6_64[(alu92+12)];
  var val42 = data6_64[(alu92+16)];
  var val43 = data6_64[(alu92+20)];
  var val44 = data6_64[(alu92+24)];
  var val45 = data6_64[(alu92+28)];
  var val46 = data6_64[(alu92+32)];
  var val47 = data6_64[(alu92+36)];
  var val48 = data6_64[(alu92+40)];
  var val49 = data6_64[(alu92+44)];
  var val50 = data6_64[(alu92+48)];
  var val51 = data6_64[(alu92+52)];
  var val52 = data6_64[(alu92+56)];
  var val53 = data6_64[(alu92+60)];
  var alu93 = (alu92+alu89);
  var alu94 = (lidx0==0);
  var alu95 = ((val34*val36)+val35);
  var alu96 = (exp2((val38*1.4426950408889634f))*val36);
  if (alu94) {
    data0_19200[alu93] = ((2.0f*(alu95+((acc1[0]+val37)*alu96*0.25f)))+-1.0f);
  }
  if (alu94) {
    data0_19200[(alu93+1200)] = ((2.0f*(alu95+((acc1[1]+val39)*alu96*0.25f)))+-1.0f);
  }
  if (alu94) {
    data0_19200[(alu93+2400)] = ((2.0f*(alu95+((acc1[2]+val40)*alu96*0.25f)))+-1.0f);
  }
  if (alu94) {
    data0_19200[(alu93+3600)] = ((2.0f*(alu95+((acc1[3]+val41)*alu96*0.25f)))+-1.0f);
  }
  if (alu94) {
    data0_19200[(alu93+4800)] = ((2.0f*(alu95+((acc1[4]+val42)*alu96*0.25f)))+-1.0f);
  }
  if (alu94) {
    data0_19200[(alu93+6000)] = ((2.0f*(alu95+((acc1[5]+val43)*alu96*0.25f)))+-1.0f);
  }
  if (alu94) {
    data0_19200[(alu93+7200)] = ((2.0f*(alu95+((acc1[6]+val44)*alu96*0.25f)))+-1.0f);
  }
  if (alu94) {
    data0_19200[(alu93+8400)] = ((2.0f*(alu95+((acc1[7]+val45)*alu96*0.25f)))+-1.0f);
  }
  if (alu94) {
    data0_19200[(alu93+9600)] = ((2.0f*(alu95+((acc1[8]+val46)*alu96*0.25f)))+-1.0f);
  }
  if (alu94) {
    data0_19200[(alu93+10800)] = ((2.0f*(alu95+((acc1[9]+val47)*alu96*0.25f)))+-1.0f);
  }
  if (alu94) {
    data0_19200[(alu93+12000)] = ((2.0f*(alu95+((acc1[10]+val48)*alu96*0.25f)))+-1.0f);
  }
  if (alu94) {
    data0_19200[(alu93+13200)] = ((2.0f*(alu95+((acc1[11]+val49)*alu96*0.25f)))+-1.0f);
  }
  if (alu94) {
    data0_19200[(alu93+14400)] = ((2.0f*(alu95+((acc1[12]+val50)*alu96*0.25f)))+-1.0f);
  }
  if (alu94) {
    data0_19200[(alu93+15600)] = ((2.0f*(alu95+((acc1[13]+val51)*alu96*0.25f)))+-1.0f);
  }
  if (alu94) {
    data0_19200[(alu93+16800)] = ((2.0f*(alu95+((acc1[14]+val52)*alu96*0.25f)))+-1.0f);
  }
  if (alu94) {
    data0_19200[(alu93+18000)] = ((2.0f*(alu95+((acc1[15]+val53)*alu96*0.25f)))+-1.0f);
  }
}`;

const r_75_32_4_8_32 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32,32>;
@group(0) @binding(1)var<storage,read_write>data0_9600:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_76800:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_76800:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_8192:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_32:array<f32>;
@compute @workgroup_size(4,8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,1>;
  var acc1: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 32 */
  var gidx1 = i32(gindex.y); /* 75 */
  var lidx0 = i32(lindex.x); /* 4 */
  var lidx1 = i32(lindex.y); /* 8 */
  var cast0 = bitcast<u32>(gidx1);
  var cast1 = bitcast<u32>(lidx0);
  acc0[0] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 32; Ridx0++) {
    var alu1 = (lidx1+bitcast<i32>((bitcast<u32>(Ridx0)<<3u)));
    var alu2 = (alu1+bitcast<i32>((cast0<<10u))+bitcast<i32>((cast1<<8u)));
    var val0 = data1_76800[alu2];
    var val1 = data2_76800[alu2];
    var val2 = data3_8192[(alu1+bitcast<i32>((bitcast<u32>(gidx0)<<8u)))];
    acc0[0] = (acc0[0]+((val0+val1)*val2));
  }
  var cast2 = bitcast<i32>((cast1<<3u));
  temp0[(lidx1+cast2)] = acc0[0];
  workgroupBarrier();
  acc1[0] = 0.0f;
  for (var Ridx104 = 0; Ridx104 < 8; Ridx104++) {
    var val3 = temp0[(cast2+Ridx104)];
    acc1[0] = (acc1[0]+val3);
  }
  var val4 = data4_32[gidx0];
  var alu10 = (lidx1==0);
  if (alu10) {
    data0_9600[(gidx0+bitcast<i32>((cast0<<7u))+bitcast<i32>((cast1<<5u)))] = (acc1[0]+val4);
  }
}`;

const r_2400_2_2 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_4800:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_9600:array<f32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,2>;
  var gidx0 = i32(gindex.x); /* 2400 */
  var cast0 = bitcast<u32>(gidx0);
  acc0[0] = (f32(-INFINITY));
  acc0[1] = (f32(-INFINITY));
  for (var Ridx0 = 0; Ridx0 < 2; Ridx0++) {
    var alu2 = (bitcast<i32>((cast0<<2u))+Ridx0);
    var val0 = data1_9600[alu2];
    var val1 = data1_9600[(alu2+2)];
    var alu3 = select(acc0[0],val0,(acc0[0]<val0));
    var alu4 = select(acc0[1],val1,(acc0[1]<val1));
    acc0[0] = alu3;
    acc0[1] = alu4;
  }
  var cast1 = bitcast<i32>((cast0<<1u));
  data0_4800[cast1] = acc0[0];
  data0_4800[(cast1+1)] = acc0[1];
}`;

const r_2400_2_2n1 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_4800:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_9600:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_4800:array<f32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,2>;
  var gidx0 = i32(gindex.x); /* 2400 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<i32>((cast0<<1u));
  var val0 = data2_4800[cast1];
  var alu0 = (cast1+1);
  var val1 = data2_4800[alu0];
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 2; Ridx0++) {
    var alu3 = (bitcast<i32>((cast0<<2u))+Ridx0);
    var val2 = data1_9600[alu3];
    var val3 = data1_9600[(alu3+2)];
    acc0[0] = (acc0[0]+exp2(((val2-val0)*1.4426950408889634f)));
    acc0[1] = (acc0[1]+exp2(((val3-val1)*1.4426950408889634f)));
  }
  data0_4800[cast1] = acc0[0];
  data0_4800[alu0] = acc0[1];
}`;

const r_150_16_16_2_2 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_76800:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_19200:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_147456:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_9600:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_4800:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_4800:array<f32>;
@compute @workgroup_size(16,2) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 150 */
  var lidx1 = i32(lindex.y); /* 2 */
  var cast0 = bitcast<u32>(gidx1);
  var cast1 = bitcast<u32>(lidx1);
  var alu0 = (gidx0+bitcast<i32>((cast0<<5u))+bitcast<i32>((cast1<<4u)));
  var val0 = data4_4800[alu0];
  var lidx0 = i32(lindex.x); /* 16 */
  var cast2 = bitcast<u32>(gidx0);
  var alu1 = ((gidx0*9216)+(lidx0*576));
  acc0[0] = 0.0f;
  for (var Ridx4 = 0; Ridx4 < 2; Ridx4++) {
    var alu3 = (bitcast<i32>((cast0<<3u))+bitcast<i32>((cast1<<2u))+bitcast<i32>((bitcast<u32>(Ridx4)<<1u))+(gidx0*1200));
    var val1 = data1_19200[alu3];
    var val2 = data1_19200[(alu3+1)];
    var alu4 = (val1+1.0f);
    var alu5 = (alu4*12.0f);
    var alu6 = (alu5+-0.5f);
    var alu7 = trunc(alu6);
    var cast3 = (i32(alu7));
    var alu8 = (val2+1.0f);
    var alu9 = (alu8*12.0f);
    var alu10 = (alu9+-0.5f);
    var alu11 = trunc(alu10);
    var cast4 = (i32(alu11));
    var alu12 = (alu7+-1.0f);
    var cast5 = (i32(alu12));
    var alu13 = (alu11+-1.0f);
    var cast6 = (i32(alu13));
    var alu14 = (alu6<alu7);
    var alu15 = select(cast3,cast5,alu14);
    var alu16 = (alu15+1);
    var cast7 = bitcast<u32>(alu15);
    var alu17 = (alu10<alu11);
    var alu18 = select(cast4,cast6,alu17);
    var alu19 = (alu18+1);
    var cast8 = bitcast<u32>(alu18);
    var alu20 = ((i32((alu11*2.3283064365386963e-10f)))-(i32(((alu11<0.0f)&(cast4!=0)))));
    var alu21 = ((i32((alu13*2.3283064365386963e-10f)))-(i32(((alu13<0.0f)&(cast6!=0)))));
    var alu22 = select(alu20,alu21,alu17);
    var alu23 = ((alu22<-1)|((alu22==-1)&(cast8<4294967295u)));
    var alu24 = select(alu19,0,alu23);
    var alu25 = select((alu22+(i32((bitcast<u32>(alu19)<cast8)))),0,alu23);
    var alu26 = ((0<alu25)|((0==alu25)&(23u<bitcast<u32>(alu24))));
    var alu27 = select(alu24,23,alu26);
    var cast9 = bitcast<u32>(alu27);
    var alu28 = ((cast9>>16u)*24u);
    var cast10 = bitcast<i32>((alu28<<16u));
    var alu29 = ((i32((alu7*2.3283064365386963e-10f)))-(i32(((alu7<0.0f)&(cast3!=0)))));
    var alu30 = ((i32((alu12*2.3283064365386963e-10f)))-(i32(((alu12<0.0f)&(cast5!=0)))));
    var alu31 = select(alu29,alu30,alu14);
    var alu32 = ((alu31<-1)|((alu31==-1)&(cast7<4294967295u)));
    var alu33 = select(alu16,0,alu32);
    var alu34 = (cast10+bitcast<i32>(((cast9&65535u)*24u)));
    var alu35 = select((alu31+(i32((bitcast<u32>(alu16)<cast7)))),0,alu32);
    var alu36 = ((0<alu35)|((0==alu35)&(23u<bitcast<u32>(alu33))));
    var alu37 = select(alu33,23,alu36);
    var cast11 = bitcast<u32>((alu34+alu37));
    var cast12 = bitcast<u32>(cast10);
    var cast13 = bitcast<u32>(alu34);
    var alu38 = select(alu25,0,alu26);
    var alu39 = (bitcast<i32>((alu28>>16u))+(i32((cast12<0u)))+(alu38*24)+(i32((cast13<cast12))));
    var alu40 = select(alu35,0,alu36);
    var alu41 = (alu39+alu40+(i32((cast11<cast13))));
    var val3 = select(0.0f, data2_147456[((i32(cast11))+alu1)], (((-1<alu41)|((-1==alu41)&(4294967295u<cast11)))&((alu41<0)|((alu41==0)&(cast11<576u)))));
    var alu42 = (alu31<0);
    var alu43 = (alu31==0);
    var alu44 = (alu42|(alu43&(cast7<0u)));
    var alu45 = select(alu15,0,alu44);
    var alu46 = select(alu31,0,alu44);
    var alu47 = ((0<alu46)|((0==alu46)&(23u<bitcast<u32>(alu45))));
    var alu48 = select(alu45,23,alu47);
    var cast14 = bitcast<u32>((alu34+alu48));
    var alu49 = select(alu46,0,alu47);
    var alu50 = (alu39+alu49+(i32((cast14<cast13))));
    var val4 = select(0.0f, data2_147456[((i32(cast14))+alu1)], (((-1<alu50)|((-1==alu50)&(4294967295u<cast14)))&((alu50<0)|((alu50==0)&(cast14<576u)))));
    var alu51 = (alu22<0);
    var alu52 = (alu22==0);
    var alu53 = (alu51|(alu52&(cast8<0u)));
    var alu54 = select(alu18,0,alu53);
    var alu55 = select(alu22,0,alu53);
    var alu56 = ((0<alu55)|((0==alu55)&(23u<bitcast<u32>(alu54))));
    var alu57 = select(alu54,23,alu56);
    var cast15 = bitcast<u32>(alu57);
    var alu58 = ((cast15>>16u)*24u);
    var cast16 = bitcast<i32>((alu58<<16u));
    var alu59 = (cast16+bitcast<i32>(((cast15&65535u)*24u)));
    var cast17 = bitcast<u32>((alu59+alu37));
    var cast18 = bitcast<u32>(cast16);
    var cast19 = bitcast<u32>(alu59);
    var alu60 = select(alu55,0,alu56);
    var alu61 = (bitcast<i32>((alu58>>16u))+(i32((cast18<0u)))+(alu60*24)+(i32((cast19<cast18))));
    var alu62 = (alu61+alu40+(i32((cast17<cast19))));
    var val5 = select(0.0f, data2_147456[((i32(cast17))+alu1)], (((-1<alu62)|((-1==alu62)&(4294967295u<cast17)))&((alu62<0)|((alu62==0)&(cast17<576u)))));
    var cast20 = bitcast<u32>((alu59+alu48));
    var alu63 = (alu61+alu49+(i32((cast20<cast19))));
    var val6 = select(0.0f, data2_147456[((i32(cast20))+alu1)], (((-1<alu63)|((-1==alu63)&(4294967295u<cast20)))&((alu63<0)|((alu63==0)&(cast20<576u)))));
    var val7 = data3_9600[(bitcast<i32>((cast2<<1u))+Ridx4+bitcast<i32>((cast0<<6u))+bitcast<i32>((cast1<<5u)))];
    var alu64 = (-1<alu31);
    var alu65 = (-1<alu22);
    var alu66 = (-1==alu31);
    var alu67 = (-1==alu22);
    var alu68 = (alu51|(alu52&(cast8<23u)));
    var alu69 = (alu65|(alu67&(4294967294u<cast8)));
    var alu70 = ((alu64|(alu66&(4294967294u<cast7)))&(alu42|(alu43&(cast7<23u))));
    var alu71 = (alu51|(alu52&(cast8<24u)));
    var alu72 = (alu65|(alu67&(4294967295u<cast8)));
    var alu73 = ((alu64|(alu66&(4294967295u<cast7)))&(alu42|(alu43&(cast7<24u))));
    var alu74 = select((((f32(alu29))*4294967296.0f)+(f32(bitcast<u32>(cast3)))),(f32(cast3)),(((alu29==0)&(-1<cast3))|((alu29==-1)&(cast3<0))));
    var alu75 = select((((f32(alu30))*4294967296.0f)+(f32(bitcast<u32>(cast5)))),(f32(cast5)),(((alu30==0)&(-1<cast5))|((alu30==-1)&(cast5<0))));
    var alu76 = select(alu74,alu75,alu14);
    var alu77 = ((alu4*-12.0f)+alu76+1.5f);
    var alu78 = select((((f32(alu20))*4294967296.0f)+(f32(bitcast<u32>(cast4)))),(f32(cast4)),(((alu20==0)&(-1<cast4))|((alu20==-1)&(cast4<0))));
    var alu79 = select((((f32(alu21))*4294967296.0f)+(f32(bitcast<u32>(cast6)))),(f32(cast6)),(((alu21==0)&(-1<cast6))|((alu21==-1)&(cast6<0))));
    var alu80 = select(alu78,alu79,alu17);
    var alu81 = ((alu8*-12.0f)+alu80+1.5f);
    var alu82 = ((alu9-alu80)+-0.5f);
    var alu83 = ((alu5-alu76)+-0.5f);
    acc0[0] = (acc0[0]+(((val6*alu77*alu81*(f32((alu73&alu72&alu71))))+(val4*alu77*alu82*(f32((alu73&alu69&alu68))))+(val5*alu83*alu81*(f32((alu70&alu72&alu71))))+(val3*alu83*alu82*(f32((alu70&alu69&alu68)))))*exp2(((val7-val0)*1.4426950408889634f))));
  }
  var val8 = data5_4800[alu0];
  data0_76800[(lidx0+bitcast<i32>((cast2<<4u))+bitcast<i32>((cast0<<9u))+bitcast<i32>((cast1<<8u)))] = (acc0[0]*(1/val8));
}`;

const r_75_16_4_16_256 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_76800:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_76800:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_76800:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_65536:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_256:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,64>;
  var gidx0 = i32(gindex.x); /* 75 */
  var lidx0 = i32(lindex.x); /* 16 */
  var cast0 = bitcast<i32>((bitcast<u32>(gidx0)<<10u));
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  acc0[4] = 0.0f;
  acc0[5] = 0.0f;
  acc0[6] = 0.0f;
  acc0[7] = 0.0f;
  acc0[8] = 0.0f;
  acc0[9] = 0.0f;
  acc0[10] = 0.0f;
  acc0[11] = 0.0f;
  acc0[12] = 0.0f;
  acc0[13] = 0.0f;
  acc0[14] = 0.0f;
  acc0[15] = 0.0f;
  acc0[16] = 0.0f;
  acc0[17] = 0.0f;
  acc0[18] = 0.0f;
  acc0[19] = 0.0f;
  acc0[20] = 0.0f;
  acc0[21] = 0.0f;
  acc0[22] = 0.0f;
  acc0[23] = 0.0f;
  acc0[24] = 0.0f;
  acc0[25] = 0.0f;
  acc0[26] = 0.0f;
  acc0[27] = 0.0f;
  acc0[28] = 0.0f;
  acc0[29] = 0.0f;
  acc0[30] = 0.0f;
  acc0[31] = 0.0f;
  acc0[32] = 0.0f;
  acc0[33] = 0.0f;
  acc0[34] = 0.0f;
  acc0[35] = 0.0f;
  acc0[36] = 0.0f;
  acc0[37] = 0.0f;
  acc0[38] = 0.0f;
  acc0[39] = 0.0f;
  acc0[40] = 0.0f;
  acc0[41] = 0.0f;
  acc0[42] = 0.0f;
  acc0[43] = 0.0f;
  acc0[44] = 0.0f;
  acc0[45] = 0.0f;
  acc0[46] = 0.0f;
  acc0[47] = 0.0f;
  acc0[48] = 0.0f;
  acc0[49] = 0.0f;
  acc0[50] = 0.0f;
  acc0[51] = 0.0f;
  acc0[52] = 0.0f;
  acc0[53] = 0.0f;
  acc0[54] = 0.0f;
  acc0[55] = 0.0f;
  acc0[56] = 0.0f;
  acc0[57] = 0.0f;
  acc0[58] = 0.0f;
  acc0[59] = 0.0f;
  acc0[60] = 0.0f;
  acc0[61] = 0.0f;
  acc0[62] = 0.0f;
  acc0[63] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 256; Ridx0++) {
    var alu64 = (cast0+Ridx0);
    var val0 = data2_76800[alu64];
    var alu65 = (bitcast<i32>((bitcast<u32>(lidx0)<<8u))+Ridx0);
    var val1 = data3_65536[alu65];
    var val2 = data3_65536[(alu65+4096)];
    var val3 = data3_65536[(alu65+8192)];
    var val4 = data3_65536[(alu65+12288)];
    var val5 = data3_65536[(alu65+16384)];
    var val6 = data3_65536[(alu65+20480)];
    var val7 = data3_65536[(alu65+24576)];
    var val8 = data3_65536[(alu65+28672)];
    var val9 = data3_65536[(alu65+32768)];
    var val10 = data3_65536[(alu65+36864)];
    var val11 = data3_65536[(alu65+40960)];
    var val12 = data3_65536[(alu65+45056)];
    var val13 = data3_65536[(alu65+49152)];
    var val14 = data3_65536[(alu65+53248)];
    var val15 = data3_65536[(alu65+57344)];
    var val16 = data3_65536[(alu65+61440)];
    var val17 = data2_76800[(alu64+256)];
    var val18 = data2_76800[(alu64+512)];
    var val19 = data2_76800[(alu64+768)];
    acc0[0] = (acc0[0]+(val0*val1));
    acc0[1] = (acc0[1]+(val0*val2));
    acc0[2] = (acc0[2]+(val0*val3));
    acc0[3] = (acc0[3]+(val0*val4));
    acc0[4] = (acc0[4]+(val0*val5));
    acc0[5] = (acc0[5]+(val0*val6));
    acc0[6] = (acc0[6]+(val0*val7));
    acc0[7] = (acc0[7]+(val0*val8));
    acc0[8] = (acc0[8]+(val0*val9));
    acc0[9] = (acc0[9]+(val0*val10));
    acc0[10] = (acc0[10]+(val0*val11));
    acc0[11] = (acc0[11]+(val0*val12));
    acc0[12] = (acc0[12]+(val0*val13));
    acc0[13] = (acc0[13]+(val0*val14));
    acc0[14] = (acc0[14]+(val0*val15));
    acc0[15] = (acc0[15]+(val0*val16));
    acc0[16] = (acc0[16]+(val17*val1));
    acc0[17] = (acc0[17]+(val17*val2));
    acc0[18] = (acc0[18]+(val17*val3));
    acc0[19] = (acc0[19]+(val17*val4));
    acc0[20] = (acc0[20]+(val17*val5));
    acc0[21] = (acc0[21]+(val17*val6));
    acc0[22] = (acc0[22]+(val17*val7));
    acc0[23] = (acc0[23]+(val17*val8));
    acc0[24] = (acc0[24]+(val17*val9));
    acc0[25] = (acc0[25]+(val17*val10));
    acc0[26] = (acc0[26]+(val17*val11));
    acc0[27] = (acc0[27]+(val17*val12));
    acc0[28] = (acc0[28]+(val17*val13));
    acc0[29] = (acc0[29]+(val17*val14));
    acc0[30] = (acc0[30]+(val17*val15));
    acc0[31] = (acc0[31]+(val17*val16));
    acc0[32] = (acc0[32]+(val18*val1));
    acc0[33] = (acc0[33]+(val18*val2));
    acc0[34] = (acc0[34]+(val18*val3));
    acc0[35] = (acc0[35]+(val18*val4));
    acc0[36] = (acc0[36]+(val18*val5));
    acc0[37] = (acc0[37]+(val18*val6));
    acc0[38] = (acc0[38]+(val18*val7));
    acc0[39] = (acc0[39]+(val18*val8));
    acc0[40] = (acc0[40]+(val18*val9));
    acc0[41] = (acc0[41]+(val18*val10));
    acc0[42] = (acc0[42]+(val18*val11));
    acc0[43] = (acc0[43]+(val18*val12));
    acc0[44] = (acc0[44]+(val18*val13));
    acc0[45] = (acc0[45]+(val18*val14));
    acc0[46] = (acc0[46]+(val18*val15));
    acc0[47] = (acc0[47]+(val18*val16));
    acc0[48] = (acc0[48]+(val19*val1));
    acc0[49] = (acc0[49]+(val19*val2));
    acc0[50] = (acc0[50]+(val19*val3));
    acc0[51] = (acc0[51]+(val19*val4));
    acc0[52] = (acc0[52]+(val19*val5));
    acc0[53] = (acc0[53]+(val19*val6));
    acc0[54] = (acc0[54]+(val19*val7));
    acc0[55] = (acc0[55]+(val19*val8));
    acc0[56] = (acc0[56]+(val19*val9));
    acc0[57] = (acc0[57]+(val19*val10));
    acc0[58] = (acc0[58]+(val19*val11));
    acc0[59] = (acc0[59]+(val19*val12));
    acc0[60] = (acc0[60]+(val19*val13));
    acc0[61] = (acc0[61]+(val19*val14));
    acc0[62] = (acc0[62]+(val19*val15));
    acc0[63] = (acc0[63]+(val19*val16));
  }
  var alu131 = (lidx0+cast0);
  var val20 = data1_76800[alu131];
  var val21 = data4_256[lidx0];
  var alu132 = (alu131+16);
  var val22 = data1_76800[alu132];
  var val23 = data4_256[(lidx0+16)];
  var alu133 = (alu131+32);
  var val24 = data1_76800[alu133];
  var val25 = data4_256[(lidx0+32)];
  var alu134 = (alu131+48);
  var val26 = data1_76800[alu134];
  var val27 = data4_256[(lidx0+48)];
  var alu135 = (alu131+64);
  var val28 = data1_76800[alu135];
  var val29 = data4_256[(lidx0+64)];
  var alu136 = (alu131+80);
  var val30 = data1_76800[alu136];
  var val31 = data4_256[(lidx0+80)];
  var alu137 = (alu131+96);
  var val32 = data1_76800[alu137];
  var val33 = data4_256[(lidx0+96)];
  var alu138 = (alu131+112);
  var val34 = data1_76800[alu138];
  var val35 = data4_256[(lidx0+112)];
  var alu139 = (alu131+128);
  var val36 = data1_76800[alu139];
  var val37 = data4_256[(lidx0+128)];
  var alu140 = (alu131+144);
  var val38 = data1_76800[alu140];
  var val39 = data4_256[(lidx0+144)];
  var alu141 = (alu131+160);
  var val40 = data1_76800[alu141];
  var val41 = data4_256[(lidx0+160)];
  var alu142 = (alu131+176);
  var val42 = data1_76800[alu142];
  var val43 = data4_256[(lidx0+176)];
  var alu143 = (alu131+192);
  var val44 = data1_76800[alu143];
  var val45 = data4_256[(lidx0+192)];
  var alu144 = (alu131+208);
  var val46 = data1_76800[alu144];
  var val47 = data4_256[(lidx0+208)];
  var alu145 = (alu131+224);
  var val48 = data1_76800[alu145];
  var val49 = data4_256[(lidx0+224)];
  var alu146 = (alu131+240);
  var val50 = data1_76800[alu146];
  var val51 = data4_256[(lidx0+240)];
  var alu147 = (alu131+256);
  var val52 = data1_76800[alu147];
  var alu148 = (alu131+272);
  var val53 = data1_76800[alu148];
  var alu149 = (alu131+288);
  var val54 = data1_76800[alu149];
  var alu150 = (alu131+304);
  var val55 = data1_76800[alu150];
  var alu151 = (alu131+320);
  var val56 = data1_76800[alu151];
  var alu152 = (alu131+336);
  var val57 = data1_76800[alu152];
  var alu153 = (alu131+352);
  var val58 = data1_76800[alu153];
  var alu154 = (alu131+368);
  var val59 = data1_76800[alu154];
  var alu155 = (alu131+384);
  var val60 = data1_76800[alu155];
  var alu156 = (alu131+400);
  var val61 = data1_76800[alu156];
  var alu157 = (alu131+416);
  var val62 = data1_76800[alu157];
  var alu158 = (alu131+432);
  var val63 = data1_76800[alu158];
  var alu159 = (alu131+448);
  var val64 = data1_76800[alu159];
  var alu160 = (alu131+464);
  var val65 = data1_76800[alu160];
  var alu161 = (alu131+480);
  var val66 = data1_76800[alu161];
  var alu162 = (alu131+496);
  var val67 = data1_76800[alu162];
  var alu163 = (alu131+512);
  var val68 = data1_76800[alu163];
  var alu164 = (alu131+528);
  var val69 = data1_76800[alu164];
  var alu165 = (alu131+544);
  var val70 = data1_76800[alu165];
  var alu166 = (alu131+560);
  var val71 = data1_76800[alu166];
  var alu167 = (alu131+576);
  var val72 = data1_76800[alu167];
  var alu168 = (alu131+592);
  var val73 = data1_76800[alu168];
  var alu169 = (alu131+608);
  var val74 = data1_76800[alu169];
  var alu170 = (alu131+624);
  var val75 = data1_76800[alu170];
  var alu171 = (alu131+640);
  var val76 = data1_76800[alu171];
  var alu172 = (alu131+656);
  var val77 = data1_76800[alu172];
  var alu173 = (alu131+672);
  var val78 = data1_76800[alu173];
  var alu174 = (alu131+688);
  var val79 = data1_76800[alu174];
  var alu175 = (alu131+704);
  var val80 = data1_76800[alu175];
  var alu176 = (alu131+720);
  var val81 = data1_76800[alu176];
  var alu177 = (alu131+736);
  var val82 = data1_76800[alu177];
  var alu178 = (alu131+752);
  var val83 = data1_76800[alu178];
  var alu179 = (alu131+768);
  var val84 = data1_76800[alu179];
  var alu180 = (alu131+784);
  var val85 = data1_76800[alu180];
  var alu181 = (alu131+800);
  var val86 = data1_76800[alu181];
  var alu182 = (alu131+816);
  var val87 = data1_76800[alu182];
  var alu183 = (alu131+832);
  var val88 = data1_76800[alu183];
  var alu184 = (alu131+848);
  var val89 = data1_76800[alu184];
  var alu185 = (alu131+864);
  var val90 = data1_76800[alu185];
  var alu186 = (alu131+880);
  var val91 = data1_76800[alu186];
  var alu187 = (alu131+896);
  var val92 = data1_76800[alu187];
  var alu188 = (alu131+912);
  var val93 = data1_76800[alu188];
  var alu189 = (alu131+928);
  var val94 = data1_76800[alu189];
  var alu190 = (alu131+944);
  var val95 = data1_76800[alu190];
  var alu191 = (alu131+960);
  var val96 = data1_76800[alu191];
  var alu192 = (alu131+976);
  var val97 = data1_76800[alu192];
  var alu193 = (alu131+992);
  var val98 = data1_76800[alu193];
  var alu194 = (alu131+1008);
  var val99 = data1_76800[alu194];
  data0_76800[alu131] = (val20+acc0[0]+val21);
  data0_76800[alu132] = (val22+acc0[1]+val23);
  data0_76800[alu133] = (val24+acc0[2]+val25);
  data0_76800[alu134] = (val26+acc0[3]+val27);
  data0_76800[alu135] = (val28+acc0[4]+val29);
  data0_76800[alu136] = (val30+acc0[5]+val31);
  data0_76800[alu137] = (val32+acc0[6]+val33);
  data0_76800[alu138] = (val34+acc0[7]+val35);
  data0_76800[alu139] = (val36+acc0[8]+val37);
  data0_76800[alu140] = (val38+acc0[9]+val39);
  data0_76800[alu141] = (val40+acc0[10]+val41);
  data0_76800[alu142] = (val42+acc0[11]+val43);
  data0_76800[alu143] = (val44+acc0[12]+val45);
  data0_76800[alu144] = (val46+acc0[13]+val47);
  data0_76800[alu145] = (val48+acc0[14]+val49);
  data0_76800[alu146] = (val50+acc0[15]+val51);
  data0_76800[alu147] = (val52+acc0[16]+val21);
  data0_76800[alu148] = (val53+acc0[17]+val23);
  data0_76800[alu149] = (val54+acc0[18]+val25);
  data0_76800[alu150] = (val55+acc0[19]+val27);
  data0_76800[alu151] = (val56+acc0[20]+val29);
  data0_76800[alu152] = (val57+acc0[21]+val31);
  data0_76800[alu153] = (val58+acc0[22]+val33);
  data0_76800[alu154] = (val59+acc0[23]+val35);
  data0_76800[alu155] = (val60+acc0[24]+val37);
  data0_76800[alu156] = (val61+acc0[25]+val39);
  data0_76800[alu157] = (val62+acc0[26]+val41);
  data0_76800[alu158] = (val63+acc0[27]+val43);
  data0_76800[alu159] = (val64+acc0[28]+val45);
  data0_76800[alu160] = (val65+acc0[29]+val47);
  data0_76800[alu161] = (val66+acc0[30]+val49);
  data0_76800[alu162] = (val67+acc0[31]+val51);
  data0_76800[alu163] = (val68+acc0[32]+val21);
  data0_76800[alu164] = (val69+acc0[33]+val23);
  data0_76800[alu165] = (val70+acc0[34]+val25);
  data0_76800[alu166] = (val71+acc0[35]+val27);
  data0_76800[alu167] = (val72+acc0[36]+val29);
  data0_76800[alu168] = (val73+acc0[37]+val31);
  data0_76800[alu169] = (val74+acc0[38]+val33);
  data0_76800[alu170] = (val75+acc0[39]+val35);
  data0_76800[alu171] = (val76+acc0[40]+val37);
  data0_76800[alu172] = (val77+acc0[41]+val39);
  data0_76800[alu173] = (val78+acc0[42]+val41);
  data0_76800[alu174] = (val79+acc0[43]+val43);
  data0_76800[alu175] = (val80+acc0[44]+val45);
  data0_76800[alu176] = (val81+acc0[45]+val47);
  data0_76800[alu177] = (val82+acc0[46]+val49);
  data0_76800[alu178] = (val83+acc0[47]+val51);
  data0_76800[alu179] = (val84+acc0[48]+val21);
  data0_76800[alu180] = (val85+acc0[49]+val23);
  data0_76800[alu181] = (val86+acc0[50]+val25);
  data0_76800[alu182] = (val87+acc0[51]+val27);
  data0_76800[alu183] = (val88+acc0[52]+val29);
  data0_76800[alu184] = (val89+acc0[53]+val31);
  data0_76800[alu185] = (val90+acc0[54]+val33);
  data0_76800[alu186] = (val91+acc0[55]+val35);
  data0_76800[alu187] = (val92+acc0[56]+val37);
  data0_76800[alu188] = (val93+acc0[57]+val39);
  data0_76800[alu189] = (val94+acc0[58]+val41);
  data0_76800[alu190] = (val95+acc0[59]+val43);
  data0_76800[alu191] = (val96+acc0[60]+val45);
  data0_76800[alu192] = (val97+acc0[61]+val47);
  data0_76800[alu193] = (val98+acc0[62]+val49);
  data0_76800[alu194] = (val99+acc0[63]+val51);
}`;

const r_15_64_16_2_5_2_2_256 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_614400:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_76800:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_524288:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_2048:array<f32>;
@compute @workgroup_size(16,2) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,20>;
  var gidx0 = i32(gindex.x); /* 64 */
  var gidx1 = i32(gindex.y); /* 15 */
  var lidx0 = i32(lindex.x); /* 16 */
  var lidx1 = i32(lindex.y); /* 2 */
  var cast0 = bitcast<u32>(gidx0);
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  acc0[4] = 0.0f;
  acc0[5] = 0.0f;
  acc0[6] = 0.0f;
  acc0[7] = 0.0f;
  acc0[8] = 0.0f;
  acc0[9] = 0.0f;
  acc0[10] = 0.0f;
  acc0[11] = 0.0f;
  acc0[12] = 0.0f;
  acc0[13] = 0.0f;
  acc0[14] = 0.0f;
  acc0[15] = 0.0f;
  acc0[16] = 0.0f;
  acc0[17] = 0.0f;
  acc0[18] = 0.0f;
  acc0[19] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 256; Ridx0++) {
    var alu20 = ((gidx1*5120)+(lidx1*1280)+Ridx0);
    var val0 = data1_76800[alu20];
    var alu21 = (bitcast<i32>((cast0<<13u))+bitcast<i32>((bitcast<u32>(lidx0)<<8u))+Ridx0);
    var val1 = data2_524288[alu21];
    var val2 = data1_76800[(alu20+2560)];
    var val3 = data2_524288[(alu21+4096)];
    var val4 = data1_76800[(alu20+256)];
    var val5 = data1_76800[(alu20+2816)];
    var val6 = data1_76800[(alu20+512)];
    var val7 = data1_76800[(alu20+3072)];
    var val8 = data1_76800[(alu20+768)];
    var val9 = data1_76800[(alu20+3328)];
    var val10 = data1_76800[(alu20+1024)];
    var val11 = data1_76800[(alu20+3584)];
    acc0[0] = (acc0[0]+(val0*val1));
    acc0[1] = (acc0[1]+(val2*val1));
    acc0[2] = (acc0[2]+(val0*val3));
    acc0[3] = (acc0[3]+(val2*val3));
    acc0[4] = (acc0[4]+(val4*val1));
    acc0[5] = (acc0[5]+(val5*val1));
    acc0[6] = (acc0[6]+(val4*val3));
    acc0[7] = (acc0[7]+(val5*val3));
    acc0[8] = (acc0[8]+(val6*val1));
    acc0[9] = (acc0[9]+(val7*val1));
    acc0[10] = (acc0[10]+(val6*val3));
    acc0[11] = (acc0[11]+(val7*val3));
    acc0[12] = (acc0[12]+(val8*val1));
    acc0[13] = (acc0[13]+(val9*val1));
    acc0[14] = (acc0[14]+(val8*val3));
    acc0[15] = (acc0[15]+(val9*val3));
    acc0[16] = (acc0[16]+(val10*val1));
    acc0[17] = (acc0[17]+(val11*val1));
    acc0[18] = (acc0[18]+(val10*val3));
    acc0[19] = (acc0[19]+(val11*val3));
  }
  var alu43 = (lidx0+bitcast<i32>((cast0<<5u)));
  var val12 = data3_2048[alu43];
  var val13 = data3_2048[(alu43+16)];
  var alu44 = (alu43+(gidx1*40960)+(lidx1*10240));
  var alu45 = (acc0[0]+val12);
  var alu46 = (acc0[1]+val12);
  var alu47 = (acc0[2]+val13);
  var alu48 = (acc0[3]+val13);
  var alu49 = (acc0[4]+val12);
  var alu50 = (acc0[5]+val12);
  var alu51 = (acc0[6]+val13);
  var alu52 = (acc0[7]+val13);
  var alu53 = (acc0[8]+val12);
  var alu54 = (acc0[9]+val12);
  var alu55 = (acc0[10]+val13);
  var alu56 = (acc0[11]+val13);
  var alu57 = (acc0[12]+val12);
  var alu58 = (acc0[13]+val12);
  var alu59 = (acc0[14]+val13);
  var alu60 = (acc0[15]+val13);
  var alu61 = (acc0[16]+val12);
  var alu62 = (acc0[17]+val12);
  var alu63 = (acc0[18]+val13);
  var alu64 = (acc0[19]+val13);
  var alu65 = select(0.0f,alu45,(0.0f<alu45));
  var alu66 = select(0.0f,alu46,(0.0f<alu46));
  var alu67 = select(0.0f,alu47,(0.0f<alu47));
  var alu68 = select(0.0f,alu48,(0.0f<alu48));
  var alu69 = select(0.0f,alu49,(0.0f<alu49));
  var alu70 = select(0.0f,alu50,(0.0f<alu50));
  var alu71 = select(0.0f,alu51,(0.0f<alu51));
  var alu72 = select(0.0f,alu52,(0.0f<alu52));
  var alu73 = select(0.0f,alu53,(0.0f<alu53));
  var alu74 = select(0.0f,alu54,(0.0f<alu54));
  var alu75 = select(0.0f,alu55,(0.0f<alu55));
  var alu76 = select(0.0f,alu56,(0.0f<alu56));
  var alu77 = select(0.0f,alu57,(0.0f<alu57));
  var alu78 = select(0.0f,alu58,(0.0f<alu58));
  var alu79 = select(0.0f,alu59,(0.0f<alu59));
  var alu80 = select(0.0f,alu60,(0.0f<alu60));
  var alu81 = select(0.0f,alu61,(0.0f<alu61));
  var alu82 = select(0.0f,alu62,(0.0f<alu62));
  var alu83 = select(0.0f,alu63,(0.0f<alu63));
  var alu84 = select(0.0f,alu64,(0.0f<alu64));
  data0_614400[alu44] = alu65;
  data0_614400[(alu44+16)] = alu67;
  data0_614400[(alu44+2048)] = alu69;
  data0_614400[(alu44+2064)] = alu71;
  data0_614400[(alu44+4096)] = alu73;
  data0_614400[(alu44+4112)] = alu75;
  data0_614400[(alu44+6144)] = alu77;
  data0_614400[(alu44+6160)] = alu79;
  data0_614400[(alu44+8192)] = alu81;
  data0_614400[(alu44+8208)] = alu83;
  data0_614400[(alu44+20480)] = alu66;
  data0_614400[(alu44+20496)] = alu68;
  data0_614400[(alu44+22528)] = alu70;
  data0_614400[(alu44+22544)] = alu72;
  data0_614400[(alu44+24576)] = alu74;
  data0_614400[(alu44+24592)] = alu76;
  data0_614400[(alu44+26624)] = alu78;
  data0_614400[(alu44+26640)] = alu80;
  data0_614400[(alu44+28672)] = alu82;
  data0_614400[(alu44+28688)] = alu84;
}`;

const r_60_8_16_5_2_2048 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_76800:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_76800:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_614400:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_524288:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_256:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,10>;
  var gidx0 = i32(gindex.x); /* 8 */
  var gidx1 = i32(gindex.y); /* 60 */
  var lidx0 = i32(lindex.x); /* 16 */
  var cast0 = bitcast<u32>(gidx0);
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  acc0[4] = 0.0f;
  acc0[5] = 0.0f;
  acc0[6] = 0.0f;
  acc0[7] = 0.0f;
  acc0[8] = 0.0f;
  acc0[9] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 2048; Ridx0++) {
    var alu10 = ((gidx1*10240)+Ridx0);
    var val0 = data2_614400[alu10];
    var alu11 = (bitcast<i32>((cast0<<16u))+bitcast<i32>((bitcast<u32>(lidx0)<<11u))+Ridx0);
    var val1 = data3_524288[alu11];
    var val2 = data3_524288[(alu11+32768)];
    var val3 = data2_614400[(alu10+2048)];
    var val4 = data2_614400[(alu10+4096)];
    var val5 = data2_614400[(alu10+6144)];
    var val6 = data2_614400[(alu10+8192)];
    acc0[0] = (acc0[0]+(val0*val1));
    acc0[1] = (acc0[1]+(val0*val2));
    acc0[2] = (acc0[2]+(val3*val1));
    acc0[3] = (acc0[3]+(val3*val2));
    acc0[4] = (acc0[4]+(val4*val1));
    acc0[5] = (acc0[5]+(val4*val2));
    acc0[6] = (acc0[6]+(val5*val1));
    acc0[7] = (acc0[7]+(val5*val2));
    acc0[8] = (acc0[8]+(val6*val1));
    acc0[9] = (acc0[9]+(val6*val2));
  }
  var alu23 = (lidx0+bitcast<i32>((cast0<<5u)));
  var alu24 = (alu23+(gidx1*1280));
  var val7 = data1_76800[alu24];
  var val8 = data4_256[alu23];
  var alu25 = (alu24+16);
  var val9 = data1_76800[alu25];
  var val10 = data4_256[(alu23+16)];
  var alu26 = (alu24+256);
  var val11 = data1_76800[alu26];
  var alu27 = (alu24+272);
  var val12 = data1_76800[alu27];
  var alu28 = (alu24+512);
  var val13 = data1_76800[alu28];
  var alu29 = (alu24+528);
  var val14 = data1_76800[alu29];
  var alu30 = (alu24+768);
  var val15 = data1_76800[alu30];
  var alu31 = (alu24+784);
  var val16 = data1_76800[alu31];
  var alu32 = (alu24+1024);
  var val17 = data1_76800[alu32];
  var alu33 = (alu24+1040);
  var val18 = data1_76800[alu33];
  data0_76800[alu24] = (val7+acc0[0]+val8);
  data0_76800[alu25] = (val9+acc0[1]+val10);
  data0_76800[alu26] = (val11+acc0[2]+val8);
  data0_76800[alu27] = (val12+acc0[3]+val10);
  data0_76800[alu28] = (val13+acc0[4]+val8);
  data0_76800[alu29] = (val14+acc0[5]+val10);
  data0_76800[alu30] = (val15+acc0[6]+val8);
  data0_76800[alu31] = (val16+acc0[7]+val10);
  data0_76800[alu32] = (val17+acc0[8]+val8);
  data0_76800[alu33] = (val18+acc0[9]+val10);
}`;

const r_60_16_16_5_256 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_76800:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_76800:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_76800:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_196608:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_768:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,5>;
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 60 */
  var lidx0 = i32(lindex.x); /* 16 */
  var cast0 = bitcast<u32>(gidx0);
  var alu0 = (gidx1*1280);
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  acc0[4] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 256; Ridx0++) {
    var alu6 = (alu0+Ridx0);
    var val0 = data1_76800[alu6];
    var val1 = data2_76800[alu6];
    var val2 = data3_196608[(bitcast<i32>((cast0<<12u))+bitcast<i32>((bitcast<u32>(lidx0)<<8u))+Ridx0)];
    var alu7 = (alu6+256);
    var val3 = data1_76800[alu7];
    var val4 = data2_76800[alu7];
    var alu8 = (alu6+512);
    var val5 = data1_76800[alu8];
    var alu9 = (alu6+768);
    var val6 = data1_76800[alu9];
    var val7 = data2_76800[alu8];
    var val8 = data2_76800[alu9];
    var alu10 = (alu6+1024);
    var val9 = data1_76800[alu10];
    var val10 = data2_76800[alu10];
    acc0[0] = (acc0[0]+((val0+val1)*val2));
    acc0[1] = (acc0[1]+((val3+val4)*val2));
    acc0[2] = (acc0[2]+((val5+val7)*val2));
    acc0[3] = (acc0[3]+((val6+val8)*val2));
    acc0[4] = (acc0[4]+((val9+val10)*val2));
  }
  var alu17 = (lidx0+bitcast<i32>((cast0<<4u)));
  var val11 = data4_768[alu17];
  var alu18 = (alu17+alu0);
  data0_76800[alu18] = (acc0[0]+val11);
  data0_76800[(alu18+256)] = (acc0[1]+val11);
  data0_76800[(alu18+512)] = (acc0[2]+val11);
  data0_76800[(alu18+768)] = (acc0[3]+val11);
  data0_76800[(alu18+1024)] = (acc0[4]+val11);
}`;

const r_60_16_16_5_256n1 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_76800:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_76800:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_76800:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_196608:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_768:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,5>;
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 60 */
  var lidx0 = i32(lindex.x); /* 16 */
  var cast0 = bitcast<u32>(gidx0);
  var alu0 = (gidx1*1280);
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  acc0[4] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 256; Ridx0++) {
    var alu6 = (alu0+Ridx0);
    var val0 = data1_76800[alu6];
    var val1 = data2_76800[alu6];
    var val2 = data3_196608[(bitcast<i32>((cast0<<12u))+bitcast<i32>((bitcast<u32>(lidx0)<<8u))+Ridx0+65536)];
    var alu7 = (alu6+256);
    var val3 = data1_76800[alu7];
    var val4 = data2_76800[alu7];
    var alu8 = (alu6+512);
    var val5 = data1_76800[alu8];
    var alu9 = (alu6+768);
    var val6 = data1_76800[alu9];
    var val7 = data2_76800[alu8];
    var val8 = data2_76800[alu9];
    var alu10 = (alu6+1024);
    var val9 = data1_76800[alu10];
    var val10 = data2_76800[alu10];
    acc0[0] = (acc0[0]+((val0+val1)*val2));
    acc0[1] = (acc0[1]+((val3+val4)*val2));
    acc0[2] = (acc0[2]+((val5+val7)*val2));
    acc0[3] = (acc0[3]+((val6+val8)*val2));
    acc0[4] = (acc0[4]+((val9+val10)*val2));
  }
  var alu17 = (lidx0+bitcast<i32>((cast0<<4u)));
  var val11 = data4_768[(alu17+256)];
  var alu18 = (alu17+alu0);
  data0_76800[alu18] = (acc0[0]+val11);
  data0_76800[(alu18+256)] = (acc0[1]+val11);
  data0_76800[(alu18+512)] = (acc0[2]+val11);
  data0_76800[(alu18+768)] = (acc0[3]+val11);
  data0_76800[(alu18+1024)] = (acc0[4]+val11);
}`;

const r_30_16_16_5_2_256 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_76800:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_76800:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_196608:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_768:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,10>;
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 30 */
  var lidx0 = i32(lindex.x); /* 16 */
  var cast0 = bitcast<u32>(gidx0);
  var alu0 = (gidx1*2560);
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  acc0[4] = 0.0f;
  acc0[5] = 0.0f;
  acc0[6] = 0.0f;
  acc0[7] = 0.0f;
  acc0[8] = 0.0f;
  acc0[9] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 256; Ridx0++) {
    var alu11 = (alu0+Ridx0);
    var val0 = data1_76800[(alu11+1280)];
    var val1 = data1_76800[(alu11+1792)];
    var val2 = data1_76800[alu11];
    var val3 = data2_196608[(bitcast<i32>((cast0<<12u))+bitcast<i32>((bitcast<u32>(lidx0)<<8u))+Ridx0+131072)];
    var val4 = data1_76800[(alu11+256)];
    var val5 = data1_76800[(alu11+1536)];
    var val6 = data1_76800[(alu11+512)];
    var val7 = data1_76800[(alu11+768)];
    var val8 = data1_76800[(alu11+2048)];
    var val9 = data1_76800[(alu11+1024)];
    var val10 = data1_76800[(alu11+2304)];
    acc0[0] = (acc0[0]+(val2*val3));
    acc0[1] = (acc0[1]+(val0*val3));
    acc0[2] = (acc0[2]+(val4*val3));
    acc0[3] = (acc0[3]+(val5*val3));
    acc0[4] = (acc0[4]+(val6*val3));
    acc0[5] = (acc0[5]+(val1*val3));
    acc0[6] = (acc0[6]+(val7*val3));
    acc0[7] = (acc0[7]+(val8*val3));
    acc0[8] = (acc0[8]+(val9*val3));
    acc0[9] = (acc0[9]+(val10*val3));
  }
  var alu23 = (lidx0+bitcast<i32>((cast0<<4u)));
  var val11 = data3_768[(alu23+512)];
  var alu24 = (alu23+alu0);
  data0_76800[alu24] = (acc0[0]+val11);
  data0_76800[(alu24+256)] = (acc0[2]+val11);
  data0_76800[(alu24+512)] = (acc0[4]+val11);
  data0_76800[(alu24+768)] = (acc0[6]+val11);
  data0_76800[(alu24+1024)] = (acc0[8]+val11);
  data0_76800[(alu24+1280)] = (acc0[1]+val11);
  data0_76800[(alu24+1536)] = (acc0[3]+val11);
  data0_76800[(alu24+1792)] = (acc0[5]+val11);
  data0_76800[(alu24+2048)] = (acc0[7]+val11);
  data0_76800[(alu24+2304)] = (acc0[9]+val11);
}`;

const E_2_300_16_16 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_153600:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_76800:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_300:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_300:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_256:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_256:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 300 */
  var lidx0 = i32(lindex.x); /* 16 */
  var alu0 = (lidx0+bitcast<i32>((bitcast<u32>(gidx0)<<4u)));
  var alu1 = (alu0+bitcast<i32>((bitcast<u32>(gidx1)<<8u)));
  var val0 = data1_76800[alu1];
  var val1 = data2_300[gidx1];
  var val2 = data3_300[gidx1];
  var val3 = data4_256[alu0];
  var val4 = data5_256[alu0];
  var gidx2 = i32(gindex.z); /* 2 */
  var alu2 = select(0.0f,(((val0-val1)*val2*val3)+val4),(gidx2<1));
  data0_153600[(alu1+(gidx2*76800))] = alu2;
}`;

const E_2_300_16_16n1 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_153600:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_76800:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_300:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_300:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_256:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_256:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 300 */
  var lidx0 = i32(lindex.x); /* 16 */
  var alu0 = (lidx0+bitcast<i32>((bitcast<u32>(gidx0)<<4u)));
  var alu1 = (alu0+bitcast<i32>((bitcast<u32>(gidx1)<<8u)));
  var val0 = data1_76800[alu1];
  var val1 = data2_300[gidx1];
  var val2 = data3_300[gidx1];
  var val3 = data4_256[alu0];
  var val4 = data5_256[alu0];
  var gidx2 = i32(gindex.z); /* 2 */
  var alu2 = select((((val0-val1)*val2*val3)+val4),0.0f,(gidx2<1));
  data0_153600[(alu1+(gidx2*76800))] = alu2;
}`;

const E_4800_32 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_153600:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_153600:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_153600:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 4800 */
  var lidx0 = i32(lindex.x); /* 32 */
  var alu0 = (lidx0+bitcast<i32>((bitcast<u32>(gidx0)<<5u)));
  var val0 = data1_153600[alu0];
  var val1 = data2_153600[alu0];
  data0_153600[alu0] = (val0+val1);
}`;

const r_300_13_7_256 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_27300:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_153600:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_23296:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_91:array<f32>;
@compute @workgroup_size(13) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,7>;
  var gidx0 = i32(gindex.x); /* 300 */
  var lidx0 = i32(lindex.x); /* 13 */
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  acc0[4] = 0.0f;
  acc0[5] = 0.0f;
  acc0[6] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 256; Ridx0++) {
    var val0 = data1_153600[(bitcast<i32>((bitcast<u32>(gidx0)<<8u))+Ridx0+76800)];
    var alu7 = (bitcast<i32>((bitcast<u32>(lidx0)<<8u))+Ridx0);
    var val1 = data2_23296[alu7];
    var val2 = data2_23296[(alu7+3328)];
    var val3 = data2_23296[(alu7+6656)];
    var val4 = data2_23296[(alu7+9984)];
    var val5 = data2_23296[(alu7+13312)];
    var val6 = data2_23296[(alu7+16640)];
    var val7 = data2_23296[(alu7+19968)];
    acc0[0] = (acc0[0]+(val0*val1));
    acc0[1] = (acc0[1]+(val0*val2));
    acc0[2] = (acc0[2]+(val0*val3));
    acc0[3] = (acc0[3]+(val0*val4));
    acc0[4] = (acc0[4]+(val0*val5));
    acc0[5] = (acc0[5]+(val0*val6));
    acc0[6] = (acc0[6]+(val0*val7));
  }
  var val8 = data3_91[lidx0];
  var val9 = data3_91[(lidx0+13)];
  var val10 = data3_91[(lidx0+26)];
  var val11 = data3_91[(lidx0+39)];
  var val12 = data3_91[(lidx0+52)];
  var val13 = data3_91[(lidx0+65)];
  var val14 = data3_91[(lidx0+78)];
  var alu16 = (lidx0+(gidx0*91));
  data0_27300[alu16] = (1/(1.0f+exp2(((acc0[0]+val8)*-1.4426950408889634f))));
  data0_27300[(alu16+13)] = (1/(1.0f+exp2(((acc0[1]+val9)*-1.4426950408889634f))));
  data0_27300[(alu16+26)] = (1/(1.0f+exp2(((acc0[2]+val10)*-1.4426950408889634f))));
  data0_27300[(alu16+39)] = (1/(1.0f+exp2(((acc0[3]+val11)*-1.4426950408889634f))));
  data0_27300[(alu16+52)] = (1/(1.0f+exp2(((acc0[4]+val12)*-1.4426950408889634f))));
  data0_27300[(alu16+65)] = (1/(1.0f+exp2(((acc0[5]+val13)*-1.4426950408889634f))));
  data0_27300[(alu16+78)] = (1/(1.0f+exp2(((acc0[6]+val14)*-1.4426950408889634f))));
}`;

const r_200_16_3_16_256 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_153600:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_153600:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_65536:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_256:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,48>;
  var gidx0 = i32(gindex.x); /* 200 */
  var lidx0 = i32(lindex.x); /* 16 */
  var alu0 = (gidx0*768);
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  acc0[4] = 0.0f;
  acc0[5] = 0.0f;
  acc0[6] = 0.0f;
  acc0[7] = 0.0f;
  acc0[8] = 0.0f;
  acc0[9] = 0.0f;
  acc0[10] = 0.0f;
  acc0[11] = 0.0f;
  acc0[12] = 0.0f;
  acc0[13] = 0.0f;
  acc0[14] = 0.0f;
  acc0[15] = 0.0f;
  acc0[16] = 0.0f;
  acc0[17] = 0.0f;
  acc0[18] = 0.0f;
  acc0[19] = 0.0f;
  acc0[20] = 0.0f;
  acc0[21] = 0.0f;
  acc0[22] = 0.0f;
  acc0[23] = 0.0f;
  acc0[24] = 0.0f;
  acc0[25] = 0.0f;
  acc0[26] = 0.0f;
  acc0[27] = 0.0f;
  acc0[28] = 0.0f;
  acc0[29] = 0.0f;
  acc0[30] = 0.0f;
  acc0[31] = 0.0f;
  acc0[32] = 0.0f;
  acc0[33] = 0.0f;
  acc0[34] = 0.0f;
  acc0[35] = 0.0f;
  acc0[36] = 0.0f;
  acc0[37] = 0.0f;
  acc0[38] = 0.0f;
  acc0[39] = 0.0f;
  acc0[40] = 0.0f;
  acc0[41] = 0.0f;
  acc0[42] = 0.0f;
  acc0[43] = 0.0f;
  acc0[44] = 0.0f;
  acc0[45] = 0.0f;
  acc0[46] = 0.0f;
  acc0[47] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 256; Ridx0++) {
    var alu49 = (alu0+Ridx0);
    var val0 = data1_153600[alu49];
    var alu50 = (bitcast<i32>((bitcast<u32>(lidx0)<<8u))+Ridx0);
    var val1 = data2_65536[alu50];
    var val2 = data2_65536[(alu50+4096)];
    var val3 = data2_65536[(alu50+8192)];
    var val4 = data2_65536[(alu50+12288)];
    var val5 = data2_65536[(alu50+16384)];
    var val6 = data2_65536[(alu50+20480)];
    var val7 = data2_65536[(alu50+24576)];
    var val8 = data2_65536[(alu50+28672)];
    var val9 = data2_65536[(alu50+32768)];
    var val10 = data2_65536[(alu50+36864)];
    var val11 = data2_65536[(alu50+40960)];
    var val12 = data2_65536[(alu50+45056)];
    var val13 = data2_65536[(alu50+49152)];
    var val14 = data2_65536[(alu50+53248)];
    var val15 = data2_65536[(alu50+57344)];
    var val16 = data2_65536[(alu50+61440)];
    var val17 = data1_153600[(alu49+256)];
    var val18 = data1_153600[(alu49+512)];
    acc0[0] = (acc0[0]+(val0*val1));
    acc0[1] = (acc0[1]+(val0*val2));
    acc0[2] = (acc0[2]+(val0*val3));
    acc0[3] = (acc0[3]+(val0*val4));
    acc0[4] = (acc0[4]+(val0*val5));
    acc0[5] = (acc0[5]+(val0*val6));
    acc0[6] = (acc0[6]+(val0*val7));
    acc0[7] = (acc0[7]+(val0*val8));
    acc0[8] = (acc0[8]+(val0*val9));
    acc0[9] = (acc0[9]+(val0*val10));
    acc0[10] = (acc0[10]+(val0*val11));
    acc0[11] = (acc0[11]+(val0*val12));
    acc0[12] = (acc0[12]+(val0*val13));
    acc0[13] = (acc0[13]+(val0*val14));
    acc0[14] = (acc0[14]+(val0*val15));
    acc0[15] = (acc0[15]+(val0*val16));
    acc0[16] = (acc0[16]+(val17*val1));
    acc0[17] = (acc0[17]+(val17*val2));
    acc0[18] = (acc0[18]+(val17*val3));
    acc0[19] = (acc0[19]+(val17*val4));
    acc0[20] = (acc0[20]+(val17*val5));
    acc0[21] = (acc0[21]+(val17*val6));
    acc0[22] = (acc0[22]+(val17*val7));
    acc0[23] = (acc0[23]+(val17*val8));
    acc0[24] = (acc0[24]+(val17*val9));
    acc0[25] = (acc0[25]+(val17*val10));
    acc0[26] = (acc0[26]+(val17*val11));
    acc0[27] = (acc0[27]+(val17*val12));
    acc0[28] = (acc0[28]+(val17*val13));
    acc0[29] = (acc0[29]+(val17*val14));
    acc0[30] = (acc0[30]+(val17*val15));
    acc0[31] = (acc0[31]+(val17*val16));
    acc0[32] = (acc0[32]+(val18*val1));
    acc0[33] = (acc0[33]+(val18*val2));
    acc0[34] = (acc0[34]+(val18*val3));
    acc0[35] = (acc0[35]+(val18*val4));
    acc0[36] = (acc0[36]+(val18*val5));
    acc0[37] = (acc0[37]+(val18*val6));
    acc0[38] = (acc0[38]+(val18*val7));
    acc0[39] = (acc0[39]+(val18*val8));
    acc0[40] = (acc0[40]+(val18*val9));
    acc0[41] = (acc0[41]+(val18*val10));
    acc0[42] = (acc0[42]+(val18*val11));
    acc0[43] = (acc0[43]+(val18*val12));
    acc0[44] = (acc0[44]+(val18*val13));
    acc0[45] = (acc0[45]+(val18*val14));
    acc0[46] = (acc0[46]+(val18*val15));
    acc0[47] = (acc0[47]+(val18*val16));
  }
  var val19 = data3_256[lidx0];
  var val20 = data3_256[(lidx0+16)];
  var val21 = data3_256[(lidx0+32)];
  var val22 = data3_256[(lidx0+48)];
  var val23 = data3_256[(lidx0+64)];
  var val24 = data3_256[(lidx0+80)];
  var val25 = data3_256[(lidx0+96)];
  var val26 = data3_256[(lidx0+112)];
  var val27 = data3_256[(lidx0+128)];
  var val28 = data3_256[(lidx0+144)];
  var val29 = data3_256[(lidx0+160)];
  var val30 = data3_256[(lidx0+176)];
  var val31 = data3_256[(lidx0+192)];
  var val32 = data3_256[(lidx0+208)];
  var val33 = data3_256[(lidx0+224)];
  var val34 = data3_256[(lidx0+240)];
  var alu100 = (lidx0+alu0);
  var alu101 = (acc0[0]+val19);
  var alu102 = (acc0[1]+val20);
  var alu103 = (acc0[2]+val21);
  var alu104 = (acc0[3]+val22);
  var alu105 = (acc0[4]+val23);
  var alu106 = (acc0[5]+val24);
  var alu107 = (acc0[6]+val25);
  var alu108 = (acc0[7]+val26);
  var alu109 = (acc0[8]+val27);
  var alu110 = (acc0[9]+val28);
  var alu111 = (acc0[10]+val29);
  var alu112 = (acc0[11]+val30);
  var alu113 = (acc0[12]+val31);
  var alu114 = (acc0[13]+val32);
  var alu115 = (acc0[14]+val33);
  var alu116 = (acc0[15]+val34);
  var alu117 = (acc0[16]+val19);
  var alu118 = (acc0[17]+val20);
  var alu119 = (acc0[18]+val21);
  var alu120 = (acc0[19]+val22);
  var alu121 = (acc0[20]+val23);
  var alu122 = (acc0[21]+val24);
  var alu123 = (acc0[22]+val25);
  var alu124 = (acc0[23]+val26);
  var alu125 = (acc0[24]+val27);
  var alu126 = (acc0[25]+val28);
  var alu127 = (acc0[26]+val29);
  var alu128 = (acc0[27]+val30);
  var alu129 = (acc0[28]+val31);
  var alu130 = (acc0[29]+val32);
  var alu131 = (acc0[30]+val33);
  var alu132 = (acc0[31]+val34);
  var alu133 = (acc0[32]+val19);
  var alu134 = (acc0[33]+val20);
  var alu135 = (acc0[34]+val21);
  var alu136 = (acc0[35]+val22);
  var alu137 = (acc0[36]+val23);
  var alu138 = (acc0[37]+val24);
  var alu139 = (acc0[38]+val25);
  var alu140 = (acc0[39]+val26);
  var alu141 = (acc0[40]+val27);
  var alu142 = (acc0[41]+val28);
  var alu143 = (acc0[42]+val29);
  var alu144 = (acc0[43]+val30);
  var alu145 = (acc0[44]+val31);
  var alu146 = (acc0[45]+val32);
  var alu147 = (acc0[46]+val33);
  var alu148 = (acc0[47]+val34);
  var alu149 = select(0.0f,alu101,(0.0f<alu101));
  var alu150 = select(0.0f,alu102,(0.0f<alu102));
  var alu151 = select(0.0f,alu103,(0.0f<alu103));
  var alu152 = select(0.0f,alu104,(0.0f<alu104));
  var alu153 = select(0.0f,alu105,(0.0f<alu105));
  var alu154 = select(0.0f,alu106,(0.0f<alu106));
  var alu155 = select(0.0f,alu107,(0.0f<alu107));
  var alu156 = select(0.0f,alu108,(0.0f<alu108));
  var alu157 = select(0.0f,alu109,(0.0f<alu109));
  var alu158 = select(0.0f,alu110,(0.0f<alu110));
  var alu159 = select(0.0f,alu111,(0.0f<alu111));
  var alu160 = select(0.0f,alu112,(0.0f<alu112));
  var alu161 = select(0.0f,alu113,(0.0f<alu113));
  var alu162 = select(0.0f,alu114,(0.0f<alu114));
  var alu163 = select(0.0f,alu115,(0.0f<alu115));
  var alu164 = select(0.0f,alu116,(0.0f<alu116));
  var alu165 = select(0.0f,alu117,(0.0f<alu117));
  var alu166 = select(0.0f,alu118,(0.0f<alu118));
  var alu167 = select(0.0f,alu119,(0.0f<alu119));
  var alu168 = select(0.0f,alu120,(0.0f<alu120));
  var alu169 = select(0.0f,alu121,(0.0f<alu121));
  var alu170 = select(0.0f,alu122,(0.0f<alu122));
  var alu171 = select(0.0f,alu123,(0.0f<alu123));
  var alu172 = select(0.0f,alu124,(0.0f<alu124));
  var alu173 = select(0.0f,alu125,(0.0f<alu125));
  var alu174 = select(0.0f,alu126,(0.0f<alu126));
  var alu175 = select(0.0f,alu127,(0.0f<alu127));
  var alu176 = select(0.0f,alu128,(0.0f<alu128));
  var alu177 = select(0.0f,alu129,(0.0f<alu129));
  var alu178 = select(0.0f,alu130,(0.0f<alu130));
  var alu179 = select(0.0f,alu131,(0.0f<alu131));
  var alu180 = select(0.0f,alu132,(0.0f<alu132));
  var alu181 = select(0.0f,alu133,(0.0f<alu133));
  var alu182 = select(0.0f,alu134,(0.0f<alu134));
  var alu183 = select(0.0f,alu135,(0.0f<alu135));
  var alu184 = select(0.0f,alu136,(0.0f<alu136));
  var alu185 = select(0.0f,alu137,(0.0f<alu137));
  var alu186 = select(0.0f,alu138,(0.0f<alu138));
  var alu187 = select(0.0f,alu139,(0.0f<alu139));
  var alu188 = select(0.0f,alu140,(0.0f<alu140));
  var alu189 = select(0.0f,alu141,(0.0f<alu141));
  var alu190 = select(0.0f,alu142,(0.0f<alu142));
  var alu191 = select(0.0f,alu143,(0.0f<alu143));
  var alu192 = select(0.0f,alu144,(0.0f<alu144));
  var alu193 = select(0.0f,alu145,(0.0f<alu145));
  var alu194 = select(0.0f,alu146,(0.0f<alu146));
  var alu195 = select(0.0f,alu147,(0.0f<alu147));
  var alu196 = select(0.0f,alu148,(0.0f<alu148));
  data0_153600[alu100] = alu149;
  data0_153600[(alu100+16)] = alu150;
  data0_153600[(alu100+32)] = alu151;
  data0_153600[(alu100+48)] = alu152;
  data0_153600[(alu100+64)] = alu153;
  data0_153600[(alu100+80)] = alu154;
  data0_153600[(alu100+96)] = alu155;
  data0_153600[(alu100+112)] = alu156;
  data0_153600[(alu100+128)] = alu157;
  data0_153600[(alu100+144)] = alu158;
  data0_153600[(alu100+160)] = alu159;
  data0_153600[(alu100+176)] = alu160;
  data0_153600[(alu100+192)] = alu161;
  data0_153600[(alu100+208)] = alu162;
  data0_153600[(alu100+224)] = alu163;
  data0_153600[(alu100+240)] = alu164;
  data0_153600[(alu100+256)] = alu165;
  data0_153600[(alu100+272)] = alu166;
  data0_153600[(alu100+288)] = alu167;
  data0_153600[(alu100+304)] = alu168;
  data0_153600[(alu100+320)] = alu169;
  data0_153600[(alu100+336)] = alu170;
  data0_153600[(alu100+352)] = alu171;
  data0_153600[(alu100+368)] = alu172;
  data0_153600[(alu100+384)] = alu173;
  data0_153600[(alu100+400)] = alu174;
  data0_153600[(alu100+416)] = alu175;
  data0_153600[(alu100+432)] = alu176;
  data0_153600[(alu100+448)] = alu177;
  data0_153600[(alu100+464)] = alu178;
  data0_153600[(alu100+480)] = alu179;
  data0_153600[(alu100+496)] = alu180;
  data0_153600[(alu100+512)] = alu181;
  data0_153600[(alu100+528)] = alu182;
  data0_153600[(alu100+544)] = alu183;
  data0_153600[(alu100+560)] = alu184;
  data0_153600[(alu100+576)] = alu185;
  data0_153600[(alu100+592)] = alu186;
  data0_153600[(alu100+608)] = alu187;
  data0_153600[(alu100+624)] = alu188;
  data0_153600[(alu100+640)] = alu189;
  data0_153600[(alu100+656)] = alu190;
  data0_153600[(alu100+672)] = alu191;
  data0_153600[(alu100+688)] = alu192;
  data0_153600[(alu100+704)] = alu193;
  data0_153600[(alu100+720)] = alu194;
  data0_153600[(alu100+736)] = alu195;
  data0_153600[(alu100+752)] = alu196;
}`;

const E_4_2_4_32_2_2_8 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_32768:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_27300:array<f32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 512 */
  var gidx1 = i32(gindex.y); /* 2 */
  var gidx2 = i32(gindex.z); /* 4 */
  var alu0 = (gidx0>>2u);
  var alu1 = (gidx0>>7u);
  var alu2 = ((alu1*37)+bitcast<i32>((bitcast<u32>((alu0&31))<<2u))+(gidx2*23));
  var alu3 = (gidx0&1);
  var alu4 = (gidx1*57);
  var alu5 = (alu2+(alu3*90)+alu4);
  var alu6 = (gidx2*1001);
  var alu7 = (alu1*91);
  var alu8 = (gidx1*455);
  var alu9 = (alu5+alu6+alu7+alu8+(alu3*-91));
  var val0 = data1_27300[(alu9+3)];
  var val1 = data1_27300[(alu9+4099)];
  var val2 = data1_27300[(alu9+8195)];
  var val3 = data1_27300[(alu9+12291)];
  var val4 = data1_27300[(alu9+16387)];
  var val5 = data1_27300[(alu9+20483)];
  var alu10 = (alu2+alu3+alu4);
  var alu11 = (alu10+alu6+alu7+alu8);
  var val6 = data1_27300[(alu11+4096)];
  var val7 = data1_27300[(alu11+8192)];
  var val8 = data1_27300[(alu11+12288)];
  var val9 = data1_27300[(alu11+16384)];
  var val10 = data1_27300[(alu11+20480)];
  var val11 = data1_27300[alu11];
  var cast0 = bitcast<u32>(gidx1);
  var cast1 = bitcast<u32>(gidx2);
  var alu12 = (alu5+9);
  var alu13 = ((gidx2*11)+alu1+(gidx1*5));
  var alu14 = ((alu12*721)>>16u);
  var alu15 = (alu13+(alu3*299)+alu14+270);
  var alu16 = ((bitcast<i32>((cast1<<8u))+alu0+bitcast<i32>((cast0<<7u)))<681);
  var val12 = select(0.0f, data1_27300[(((alu15-(300*((alu15*437)>>17u)))*91)+(alu12-(91*alu14)))], alu16);
  var alu17 = (alu10+6);
  var alu18 = ((alu17*361)>>15u);
  var alu19 = (alu13+alu18+270);
  var val13 = select(0.0f, data1_27300[(((alu19-(300*((alu19*219)>>16u)))*91)+(alu17-(91*alu18)))], alu16);
  var alu20 = (gidx0+bitcast<i32>((cast1<<10u))+bitcast<i32>((cast0<<9u)));
  var alu21 = select(0.0f,1.0f,alu16);
  var alu22 = select((f32(-INFINITY)),0.0f,(alu21!=0.0f));
  var alu23 = (((gidx0>>1u)&1)<1);
  var alu24 = select(0.0f,val6,alu23);
  var alu25 = select(val1,0.0f,alu23);
  var alu26 = select(0.0f,val7,alu23);
  var alu27 = select(val2,0.0f,alu23);
  var alu28 = select(0.0f,val8,alu23);
  var alu29 = select(val3,0.0f,alu23);
  var alu30 = select(0.0f,val9,alu23);
  var alu31 = select(val4,0.0f,alu23);
  var alu32 = select(0.0f,val10,alu23);
  var alu33 = select(val5,0.0f,alu23);
  var alu34 = select(0.0f,val11,alu23);
  var alu35 = select(val0,0.0f,alu23);
  var alu36 = select(0.0f,(val13+alu22),alu23);
  var alu37 = select((val12+alu22),0.0f,alu23);
  data0_32768[alu20] = (alu34+alu35);
  data0_32768[(alu20+4096)] = (alu24+alu25);
  data0_32768[(alu20+8192)] = (alu26+alu27);
  data0_32768[(alu20+12288)] = (alu28+alu29);
  data0_32768[(alu20+16384)] = (alu30+alu31);
  data0_32768[(alu20+20480)] = (alu32+alu33);
  data0_32768[(alu20+24576)] = (alu36+alu37);
  data0_32768[(alu20+28672)] = (f32(-INFINITY));
}`;

const r_1950_28_7_2_975 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<i32,392>;
@group(0) @binding(1)var<storage,read_write>data0_27300:array<i32>;
@group(0) @binding(2)var<storage,read_write>data1_27300:array<f32>;
@compute @workgroup_size(28) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<i32,14>;
  var acc1: array<i32,14>;
  var gidx0 = i32(gindex.x); /* 1950 */
  var alu0 = (gidx0*14);
  var alu1 = (alu0+1);
  var val0 = data1_27300[alu1];
  var alu2 = (alu0+2);
  var val1 = data1_27300[alu2];
  var alu3 = (alu0+3);
  var val2 = data1_27300[alu3];
  var alu4 = (alu0+4);
  var val3 = data1_27300[alu4];
  var alu5 = (alu0+5);
  var val4 = data1_27300[alu5];
  var alu6 = (alu0+6);
  var val5 = data1_27300[alu6];
  var alu7 = (alu0+7);
  var val6 = data1_27300[alu7];
  var alu8 = (alu0+8);
  var val7 = data1_27300[alu8];
  var alu9 = (alu0+9);
  var val8 = data1_27300[alu9];
  var alu10 = (alu0+10);
  var val9 = data1_27300[alu10];
  var alu11 = (alu0+11);
  var val10 = data1_27300[alu11];
  var alu12 = (alu0+12);
  var val11 = data1_27300[alu12];
  var alu13 = (alu0+13);
  var val12 = data1_27300[alu13];
  var val13 = data1_27300[alu0];
  var lidx0 = i32(lindex.x); /* 28 */
  acc0[0] = 0;
  acc0[1] = 0;
  acc0[2] = 0;
  acc0[3] = 0;
  acc0[4] = 0;
  acc0[5] = 0;
  acc0[6] = 0;
  acc0[7] = 0;
  acc0[8] = 0;
  acc0[9] = 0;
  acc0[10] = 0;
  acc0[11] = 0;
  acc0[12] = 0;
  acc0[13] = 0;
  for (var Ridx0 = 0; Ridx0 < 975; Ridx0++) {
    var alu28 = ((lidx0*975)+Ridx0);
    var val14 = data1_27300[alu28];
    acc0[0] = (acc0[0]+(i32(((alu28<alu1)&(val14==val13)))));
    acc0[1] = (acc0[1]+(i32(((alu28<alu8)&(val14==val6)))));
    acc0[2] = (acc0[2]+(i32(((alu28<alu2)&(val14==val0)))));
    acc0[3] = (acc0[3]+(i32(((alu28<alu9)&(val14==val7)))));
    acc0[4] = (acc0[4]+(i32(((alu28<alu3)&(val14==val1)))));
    acc0[5] = (acc0[5]+(i32(((alu28<alu10)&(val14==val8)))));
    acc0[6] = (acc0[6]+(i32(((alu28<alu4)&(val14==val2)))));
    acc0[7] = (acc0[7]+(i32(((alu28<alu11)&(val14==val9)))));
    acc0[8] = (acc0[8]+(i32(((alu28<alu5)&(val14==val3)))));
    acc0[9] = (acc0[9]+(i32(((alu28<alu12)&(val14==val10)))));
    acc0[10] = (acc0[10]+(i32(((alu28<alu6)&(val14==val4)))));
    acc0[11] = (acc0[11]+(i32(((alu28<alu13)&(val14==val11)))));
    acc0[12] = (acc0[12]+(i32(((alu28<alu7)&(val14==val5)))));
    acc0[13] = (acc0[13]+(i32(((alu28<(alu0+14))&(val14==val12)))));
  }
  var alu44 = (lidx0*14);
  temp0[(alu44+1)] = acc0[1];
  temp0[(alu44+2)] = acc0[2];
  temp0[(alu44+3)] = acc0[3];
  temp0[(alu44+4)] = acc0[4];
  temp0[(alu44+5)] = acc0[5];
  temp0[(alu44+6)] = acc0[6];
  temp0[(alu44+7)] = acc0[7];
  temp0[(alu44+8)] = acc0[8];
  temp0[(alu44+9)] = acc0[9];
  temp0[(alu44+10)] = acc0[10];
  temp0[(alu44+11)] = acc0[11];
  temp0[(alu44+12)] = acc0[12];
  temp0[(alu44+13)] = acc0[13];
  temp0[alu44] = acc0[0];
  workgroupBarrier();
  acc1[0] = 0;
  acc1[1] = 0;
  acc1[2] = 0;
  acc1[3] = 0;
  acc1[4] = 0;
  acc1[5] = 0;
  acc1[6] = 0;
  acc1[7] = 0;
  acc1[8] = 0;
  acc1[9] = 0;
  acc1[10] = 0;
  acc1[11] = 0;
  acc1[12] = 0;
  acc1[13] = 0;
  for (var Ridx102 = 0; Ridx102 < 28; Ridx102++) {
    var alu74 = (Ridx102*14);
    var val15 = temp0[(alu74+5)];
    var val16 = temp0[alu74];
    var val17 = temp0[(alu74+1)];
    var val18 = temp0[(alu74+2)];
    var val19 = temp0[(alu74+3)];
    var val20 = temp0[(alu74+4)];
    var val21 = temp0[(alu74+6)];
    var val22 = temp0[(alu74+7)];
    var val23 = temp0[(alu74+8)];
    var val24 = temp0[(alu74+9)];
    var val25 = temp0[(alu74+10)];
    var val26 = temp0[(alu74+11)];
    var val27 = temp0[(alu74+12)];
    var val28 = temp0[(alu74+13)];
    acc1[0] = (acc1[0]+val16);
    acc1[1] = (acc1[1]+val17);
    acc1[2] = (acc1[2]+val18);
    acc1[3] = (acc1[3]+val19);
    acc1[4] = (acc1[4]+val20);
    acc1[5] = (acc1[5]+val15);
    acc1[6] = (acc1[6]+val21);
    acc1[7] = (acc1[7]+val22);
    acc1[8] = (acc1[8]+val23);
    acc1[9] = (acc1[9]+val24);
    acc1[10] = (acc1[10]+val25);
    acc1[11] = (acc1[11]+val26);
    acc1[12] = (acc1[12]+val27);
    acc1[13] = (acc1[13]+val28);
  }
  var alu90 = (lidx0==0);
  if (alu90) {
    data0_27300[alu1] = acc1[2];
  }
  if (alu90) {
    data0_27300[alu2] = acc1[4];
  }
  if (alu90) {
    data0_27300[alu3] = acc1[6];
  }
  if (alu90) {
    data0_27300[alu4] = acc1[8];
  }
  if (alu90) {
    data0_27300[alu5] = acc1[10];
  }
  if (alu90) {
    data0_27300[alu6] = acc1[12];
  }
  if (alu90) {
    data0_27300[alu7] = acc1[1];
  }
  if (alu90) {
    data0_27300[alu8] = acc1[3];
  }
  if (alu90) {
    data0_27300[alu9] = acc1[5];
  }
  if (alu90) {
    data0_27300[alu10] = acc1[7];
  }
  if (alu90) {
    data0_27300[alu11] = acc1[9];
  }
  if (alu90) {
    data0_27300[alu12] = acc1[11];
  }
  if (alu90) {
    data0_27300[alu13] = acc1[13];
  }
  if (alu90) {
    data0_27300[alu0] = acc1[0];
  }
}`;

const E_1024_2_16 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_32768:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_32768:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx1 = i32(gindex.y); /* 1024 */
  var lidx0 = i32(lindex.x); /* 16 */
  var alu0 = (bitcast<i32>((bitcast<u32>(gidx1)<<5u))+bitcast<i32>((bitcast<u32>(lidx0)<<1u)));
  var val0 = data1_32768[alu0];
  var val1 = data1_32768[(alu0+1)];
  var gidx0 = i32(gindex.x); /* 2 */
  var alu1 = -val0;
  var alu2 = -val1;
  var alu3 = select(alu1,alu2,(alu1<alu2));
  var alu4 = (gidx0<1);
  var alu5 = select(val0,val1,(val0<val1));
  var alu6 = select(0.0f,alu5,alu4);
  var alu7 = select(-alu3,0.0f,alu4);
  data0_32768[(gidx0+alu0)] = (alu6+alu7);
}`;

const r_75_4_8_256 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_2400:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_153600:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_1024:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_4:array<f32>;
@compute @workgroup_size(8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 75 */
  var lidx0 = i32(lindex.x); /* 8 */
  var cast0 = bitcast<u32>(gidx1);
  var cast1 = bitcast<u32>(lidx0);
  acc0[0] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 256; Ridx0++) {
    var val0 = data1_153600[(bitcast<i32>((cast0<<11u))+bitcast<i32>((cast1<<8u))+Ridx0)];
    var val1 = data2_1024[(bitcast<i32>((bitcast<u32>(gidx0)<<8u))+Ridx0)];
    acc0[0] = (acc0[0]+(val0*val1));
  }
  var val2 = data3_4[gidx0];
  data0_2400[(gidx0+bitcast<i32>((cast0<<5u))+bitcast<i32>((cast1<<2u)))] = (acc0[0]+val2);
}`;

const E_512_2_2_2_8 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_32768:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_32768:array<f32>;
@compute @workgroup_size(8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx2 = i32(gindex.z); /* 512 */
  var lidx0 = i32(lindex.x); /* 8 */
  var cast0 = bitcast<u32>(gidx2);
  var cast1 = bitcast<u32>(lidx0);
  var alu0 = (gidx0>>1u);
  var alu1 = -alu0;
  var alu2 = (gidx0&1);
  var val0 = select(0.0f, data1_32768[(bitcast<i32>((bitcast<u32>(((bitcast<i32>((cast0<<5u))+bitcast<i32>((cast1<<2u))+alu0+3)&16383))<<1u))+alu2)], (-1<alu1));
  var cast2 = bitcast<i32>((cast0<<6u));
  var cast3 = bitcast<i32>((cast1<<3u));
  var alu3 = (cast2+cast3);
  var val1 = data1_32768[(alu3+alu2)];
  var alu4 = (alu3-alu2);
  var val2 = data1_32768[(alu4+3)];
  var val3 = data1_32768[(alu4+5)];
  var gidx1 = i32(gindex.y); /* 2 */
  var alu5 = (gidx0<2);
  var alu6 = select(0.0f,val1,alu5);
  var alu7 = select(val2,0.0f,alu5);
  var alu8 = select(0.0f,val3,(alu1<0));
  var alu9 = (gidx1<1);
  var alu10 = select(0.0f,(alu6+alu7),alu9);
  var alu11 = select((alu8+val0),0.0f,alu9);
  data0_32768[(gidx0+cast2+cast3+bitcast<i32>((bitcast<u32>(gidx1)<<2u)))] = (alu10+alu11);
}`;

const E_2048_2_2_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_32768:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_32768:array<f32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 2 */
  var gidx2 = i32(gindex.z); /* 2048 */
  var cast0 = bitcast<i32>((bitcast<u32>(gidx2)<<4u));
  var alu0 = (gidx0+cast0);
  var val0 = data1_32768[alu0];
  var val1 = data1_32768[(alu0+2)];
  var val2 = data1_32768[(alu0+4)];
  var val3 = data1_32768[(alu0+6)];
  var val4 = data1_32768[(alu0+8)];
  var val5 = data1_32768[(alu0+10)];
  var val6 = data1_32768[(alu0+12)];
  var val7 = data1_32768[(alu0+14)];
  var gidx1 = i32(gindex.y); /* 2 */
  var alu1 = (gidx0+bitcast<i32>((bitcast<u32>(gidx1)<<1u))+cast0);
  var alu2 = -val0;
  var alu3 = -val1;
  var alu4 = -val2;
  var alu5 = -val3;
  var alu6 = -val4;
  var alu7 = -val5;
  var alu8 = -val6;
  var alu9 = -val7;
  var alu10 = select(alu2,alu3,(alu2<alu3));
  var alu11 = select(alu4,alu5,(alu4<alu5));
  var alu12 = select(alu6,alu7,(alu6<alu7));
  var alu13 = select(alu8,alu9,(alu8<alu9));
  var alu14 = (gidx1<1);
  var alu15 = select(val0,val1,(val0<val1));
  var alu16 = select(0.0f,alu15,alu14);
  var alu17 = select(-alu10,0.0f,alu14);
  var alu18 = select(val2,val3,(val2<val3));
  var alu19 = select(0.0f,alu18,alu14);
  var alu20 = select(-alu11,0.0f,alu14);
  var alu21 = select(val4,val5,(val4<val5));
  var alu22 = select(0.0f,alu21,alu14);
  var alu23 = select(-alu12,0.0f,alu14);
  var alu24 = select(val6,val7,(val6<val7));
  var alu25 = select(0.0f,alu24,alu14);
  var alu26 = select(-alu13,0.0f,alu14);
  data0_32768[alu1] = (alu16+alu17);
  data0_32768[(alu1+4)] = (alu19+alu20);
  data0_32768[(alu1+8)] = (alu22+alu23);
  data0_32768[(alu1+12)] = (alu25+alu26);
}`;

const E_2048_2_2_4n1 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_32768:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_32768:array<f32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx2 = i32(gindex.z); /* 2048 */
  var cast0 = bitcast<u32>(gidx2);
  var cast1 = bitcast<i32>((cast0<<4u));
  var val0 = data1_32768[cast1];
  var gidx0 = i32(gindex.x); /* 2 */
  var cast2 = bitcast<i32>((bitcast<u32>(((gidx0+bitcast<i32>((cast0<<2u))+3)&8191))<<2u));
  var alu0 = -gidx0;
  var alu1 = (-1<alu0);
  var val1 = select(0.0f, data1_32768[cast2], alu1);
  var val2 = data1_32768[(cast1+1)];
  var val3 = data1_32768[(cast1+2)];
  var val4 = data1_32768[(cast1+3)];
  var val5 = data1_32768[(cast1+4)];
  var val6 = data1_32768[(cast1+5)];
  var val7 = data1_32768[(cast1+6)];
  var val8 = data1_32768[(cast1+7)];
  var val9 = data1_32768[(cast1+8)];
  var val10 = data1_32768[(cast1+9)];
  var val11 = data1_32768[(cast1+10)];
  var val12 = data1_32768[(cast1+11)];
  var val13 = select(0.0f, data1_32768[(cast2+1)], alu1);
  var val14 = select(0.0f, data1_32768[(cast2+2)], alu1);
  var val15 = select(0.0f, data1_32768[(cast2+3)], alu1);
  var gidx1 = i32(gindex.y); /* 2 */
  var alu2 = (bitcast<i32>((bitcast<u32>(gidx0)<<2u))+cast1+bitcast<i32>((bitcast<u32>(gidx1)<<3u)));
  var alu3 = (gidx0<1);
  var alu4 = select(0.0f,val0,alu3);
  var alu5 = select(val8,0.0f,alu3);
  var alu6 = select(0.0f,val2,alu3);
  var alu7 = select(val7,0.0f,alu3);
  var alu8 = select(0.0f,val3,alu3);
  var alu9 = select(val6,0.0f,alu3);
  var alu10 = select(0.0f,val4,alu3);
  var alu11 = select(val5,0.0f,alu3);
  var alu12 = (alu0<0);
  var alu13 = select(0.0f,val12,alu12);
  var alu14 = (gidx1<1);
  var alu15 = select(0.0f,(alu4+alu5),alu14);
  var alu16 = select((alu13+val1),0.0f,alu14);
  var alu17 = select(0.0f,val11,alu12);
  var alu18 = select(0.0f,(alu6+alu7),alu14);
  var alu19 = select((alu17+val13),0.0f,alu14);
  var alu20 = select(0.0f,val10,alu12);
  var alu21 = select(0.0f,(alu8+alu9),alu14);
  var alu22 = select((alu20+val14),0.0f,alu14);
  var alu23 = select(0.0f,val9,alu12);
  var alu24 = select(0.0f,(alu10+alu11),alu14);
  var alu25 = select((alu23+val15),0.0f,alu14);
  data0_32768[alu2] = (alu15+alu16);
  data0_32768[(alu2+1)] = (alu18+alu19);
  data0_32768[(alu2+2)] = (alu21+alu22);
  data0_32768[(alu2+3)] = (alu24+alu25);
}`;

const E_1024_2_4_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_32768:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_32768:array<f32>;
@compute @workgroup_size(4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx2 = i32(gindex.z); /* 1024 */
  var lidx0 = i32(lindex.x); /* 4 */
  var alu0 = (gidx0+bitcast<i32>((bitcast<u32>(gidx2)<<5u))+bitcast<i32>((bitcast<u32>(lidx0)<<3u)));
  var val0 = data1_32768[alu0];
  var val1 = data1_32768[(alu0+4)];
  var gidx1 = i32(gindex.y); /* 2 */
  var alu1 = -val0;
  var alu2 = -val1;
  var alu3 = select(alu1,alu2,(alu1<alu2));
  var alu4 = (gidx1<1);
  var alu5 = select(val0,val1,(val0<val1));
  var alu6 = select(0.0f,alu5,alu4);
  var alu7 = select(-alu3,0.0f,alu4);
  data0_32768[(alu0+bitcast<i32>((bitcast<u32>(gidx1)<<2u)))] = (alu6+alu7);
}`;

const E_1024_2_2_2_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_32768:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_32768:array<f32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx2 = i32(gindex.z); /* 1024 */
  var cast0 = bitcast<u32>(gidx2);
  var cast1 = bitcast<i32>((cast0<<5u));
  var alu0 = (gidx0&1);
  var cast2 = bitcast<i32>((bitcast<u32>(alu0)<<2u));
  var alu1 = (cast1+cast2);
  var val0 = data1_32768[alu1];
  var alu2 = (gidx0>>1u);
  var alu3 = -alu2;
  var alu4 = (cast2+bitcast<i32>((bitcast<u32>(((bitcast<i32>((cast0<<2u))+alu2+3)&4095))<<3u)));
  var alu5 = (-1<alu3);
  var val1 = select(0.0f, data1_32768[alu4], alu5);
  var val2 = data1_32768[(alu1+1)];
  var val3 = data1_32768[(alu1+2)];
  var val4 = data1_32768[(alu1+3)];
  var alu6 = (cast1+(alu0*-4));
  var val5 = data1_32768[(alu6+12)];
  var val6 = data1_32768[(alu6+13)];
  var val7 = data1_32768[(alu6+14)];
  var val8 = data1_32768[(alu6+15)];
  var val9 = data1_32768[(alu6+20)];
  var val10 = data1_32768[(alu6+21)];
  var val11 = data1_32768[(alu6+22)];
  var val12 = data1_32768[(alu6+23)];
  var val13 = select(0.0f, data1_32768[(alu4+1)], alu5);
  var val14 = select(0.0f, data1_32768[(alu4+2)], alu5);
  var val15 = select(0.0f, data1_32768[(alu4+3)], alu5);
  var gidx1 = i32(gindex.y); /* 2 */
  var alu7 = (bitcast<i32>((bitcast<u32>(gidx0)<<2u))+cast1+bitcast<i32>((bitcast<u32>(gidx1)<<4u)));
  var alu8 = (gidx0<2);
  var alu9 = select(0.0f,val0,alu8);
  var alu10 = select(val8,0.0f,alu8);
  var alu11 = select(0.0f,val2,alu8);
  var alu12 = select(val7,0.0f,alu8);
  var alu13 = select(0.0f,val3,alu8);
  var alu14 = select(val6,0.0f,alu8);
  var alu15 = select(0.0f,val4,alu8);
  var alu16 = select(val5,0.0f,alu8);
  var alu17 = (alu3<0);
  var alu18 = select(0.0f,val12,alu17);
  var alu19 = (gidx1<1);
  var alu20 = select(0.0f,(alu9+alu10),alu19);
  var alu21 = select((alu18+val1),0.0f,alu19);
  var alu22 = select(0.0f,val11,alu17);
  var alu23 = select(0.0f,(alu11+alu12),alu19);
  var alu24 = select((alu22+val13),0.0f,alu19);
  var alu25 = select(0.0f,val10,alu17);
  var alu26 = select(0.0f,(alu13+alu14),alu19);
  var alu27 = select((alu25+val14),0.0f,alu19);
  var alu28 = select(0.0f,val9,alu17);
  var alu29 = select(0.0f,(alu15+alu16),alu19);
  var alu30 = select((alu28+val15),0.0f,alu19);
  data0_32768[alu7] = (alu20+alu21);
  data0_32768[(alu7+1)] = (alu23+alu24);
  data0_32768[(alu7+2)] = (alu26+alu27);
  data0_32768[(alu7+3)] = (alu29+alu30);
}`;

const E_2048_2_8 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_32768:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_32768:array<f32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx1 = i32(gindex.y); /* 2048 */
  var cast0 = bitcast<i32>((bitcast<u32>(gidx1)<<4u));
  var val0 = data1_32768[cast0];
  var val1 = data1_32768[(cast0+1)];
  var val2 = data1_32768[(cast0+2)];
  var val3 = data1_32768[(cast0+3)];
  var val4 = data1_32768[(cast0+4)];
  var val5 = data1_32768[(cast0+5)];
  var val6 = data1_32768[(cast0+6)];
  var val7 = data1_32768[(cast0+7)];
  var val8 = data1_32768[(cast0+8)];
  var val9 = data1_32768[(cast0+9)];
  var val10 = data1_32768[(cast0+10)];
  var val11 = data1_32768[(cast0+11)];
  var val12 = data1_32768[(cast0+12)];
  var val13 = data1_32768[(cast0+13)];
  var val14 = data1_32768[(cast0+14)];
  var val15 = data1_32768[(cast0+15)];
  var gidx0 = i32(gindex.x); /* 2 */
  var alu0 = (bitcast<i32>((bitcast<u32>(gidx0)<<3u))+cast0);
  var alu1 = -val0;
  var alu2 = -val1;
  var alu3 = -val2;
  var alu4 = -val3;
  var alu5 = -val4;
  var alu6 = -val5;
  var alu7 = -val6;
  var alu8 = -val7;
  var alu9 = -val8;
  var alu10 = -val9;
  var alu11 = -val10;
  var alu12 = -val11;
  var alu13 = -val12;
  var alu14 = -val13;
  var alu15 = -val14;
  var alu16 = -val15;
  var alu17 = select(alu1,alu9,(alu1<alu9));
  var alu18 = select(alu2,alu10,(alu2<alu10));
  var alu19 = select(alu3,alu11,(alu3<alu11));
  var alu20 = select(alu4,alu12,(alu4<alu12));
  var alu21 = select(alu5,alu13,(alu5<alu13));
  var alu22 = select(alu6,alu14,(alu6<alu14));
  var alu23 = select(alu7,alu15,(alu7<alu15));
  var alu24 = select(alu8,alu16,(alu8<alu16));
  var alu25 = (gidx0<1);
  var alu26 = select(val0,val8,(val0<val8));
  var alu27 = select(0.0f,alu26,alu25);
  var alu28 = select(-alu17,0.0f,alu25);
  var alu29 = select(val1,val9,(val1<val9));
  var alu30 = select(0.0f,alu29,alu25);
  var alu31 = select(-alu18,0.0f,alu25);
  var alu32 = select(val2,val10,(val2<val10));
  var alu33 = select(0.0f,alu32,alu25);
  var alu34 = select(-alu19,0.0f,alu25);
  var alu35 = select(val3,val11,(val3<val11));
  var alu36 = select(0.0f,alu35,alu25);
  var alu37 = select(-alu20,0.0f,alu25);
  var alu38 = select(val4,val12,(val4<val12));
  var alu39 = select(0.0f,alu38,alu25);
  var alu40 = select(-alu21,0.0f,alu25);
  var alu41 = select(val5,val13,(val5<val13));
  var alu42 = select(0.0f,alu41,alu25);
  var alu43 = select(-alu22,0.0f,alu25);
  var alu44 = select(val6,val14,(val6<val14));
  var alu45 = select(0.0f,alu44,alu25);
  var alu46 = select(-alu23,0.0f,alu25);
  var alu47 = select(val7,val15,(val7<val15));
  var alu48 = select(0.0f,alu47,alu25);
  var alu49 = select(-alu24,0.0f,alu25);
  data0_32768[alu0] = (alu27+alu28);
  data0_32768[(alu0+1)] = (alu30+alu31);
  data0_32768[(alu0+2)] = (alu33+alu34);
  data0_32768[(alu0+3)] = (alu36+alu37);
  data0_32768[(alu0+4)] = (alu39+alu40);
  data0_32768[(alu0+5)] = (alu42+alu43);
  data0_32768[(alu0+6)] = (alu45+alu46);
  data0_32768[(alu0+7)] = (alu48+alu49);
}`;

const E_512_2_16_2 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_32768:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_32768:array<f32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx2 = i32(gindex.z); /* 512 */
  var cast0 = bitcast<u32>(gidx2);
  var cast1 = bitcast<i32>((cast0<<6u));
  var alu0 = (gidx0+cast1);
  var val0 = data1_32768[alu0];
  var gidx1 = i32(gindex.y); /* 2 */
  var alu1 = -gidx1;
  var val1 = select(0.0f, data1_32768[(gidx0+bitcast<i32>((bitcast<u32>(((gidx1+bitcast<i32>((cast0<<2u))+3)&2047))<<4u)))], (-1<alu1));
  var alu2 = (cast1-gidx0);
  var val2 = data1_32768[(alu2+31)];
  var val3 = data1_32768[(alu2+47)];
  var alu3 = (alu0+bitcast<i32>((bitcast<u32>(gidx1)<<4u)));
  var alu4 = (gidx1<1);
  var alu5 = select(0.0f,val0,alu4);
  var alu6 = select(val2,0.0f,alu4);
  var alu7 = select(0.0f,val3,(alu1<0));
  data0_32768[alu3] = (alu5+alu6);
  data0_32768[(alu3+32)] = (alu7+val1);
}`;

const E_1024_2_4_4n1 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_32768:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_32768:array<f32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx2 = i32(gindex.z); /* 1024 */
  var alu0 = (bitcast<i32>((bitcast<u32>(gidx0)<<2u))+bitcast<i32>((bitcast<u32>(gidx2)<<5u)));
  var val0 = data1_32768[alu0];
  var val1 = data1_32768[(alu0+1)];
  var val2 = data1_32768[(alu0+2)];
  var val3 = data1_32768[(alu0+3)];
  var val4 = data1_32768[(alu0+16)];
  var val5 = data1_32768[(alu0+17)];
  var val6 = data1_32768[(alu0+18)];
  var val7 = data1_32768[(alu0+19)];
  var gidx1 = i32(gindex.y); /* 2 */
  var alu1 = (alu0+bitcast<i32>((bitcast<u32>(gidx1)<<4u)));
  var alu2 = -val0;
  var alu3 = -val1;
  var alu4 = -val2;
  var alu5 = -val3;
  var alu6 = -val4;
  var alu7 = -val5;
  var alu8 = -val6;
  var alu9 = -val7;
  var alu10 = select(alu2,alu6,(alu2<alu6));
  var alu11 = select(alu3,alu7,(alu3<alu7));
  var alu12 = select(alu4,alu8,(alu4<alu8));
  var alu13 = select(alu5,alu9,(alu5<alu9));
  var alu14 = (gidx1<1);
  var alu15 = select(val0,val4,(val0<val4));
  var alu16 = select(0.0f,alu15,alu14);
  var alu17 = select(-alu10,0.0f,alu14);
  var alu18 = select(val1,val5,(val1<val5));
  var alu19 = select(0.0f,alu18,alu14);
  var alu20 = select(-alu11,0.0f,alu14);
  var alu21 = select(val2,val6,(val2<val6));
  var alu22 = select(0.0f,alu21,alu14);
  var alu23 = select(-alu12,0.0f,alu14);
  var alu24 = select(val3,val7,(val3<val7));
  var alu25 = select(0.0f,alu24,alu14);
  var alu26 = select(-alu13,0.0f,alu14);
  data0_32768[alu1] = (alu16+alu17);
  data0_32768[(alu1+1)] = (alu19+alu20);
  data0_32768[(alu1+2)] = (alu22+alu23);
  data0_32768[(alu1+3)] = (alu25+alu26);
}`;

const E_32_2_2_32_8 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_32768:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_32768:array<f32>;
@compute @workgroup_size(8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 64 */
  var gidx2 = i32(gindex.z); /* 32 */
  var lidx0 = i32(lindex.x); /* 8 */
  var cast0 = bitcast<u32>(gidx2);
  var cast1 = bitcast<u32>(lidx0);
  var alu0 = (gidx0>>5u);
  var alu1 = -alu0;
  var alu2 = (gidx0&31);
  var val0 = select(0.0f, data1_32768[(bitcast<i32>((bitcast<u32>(((bitcast<i32>((cast0<<5u))+bitcast<i32>((cast1<<2u))+alu0+3)&1023))<<5u))+alu2)], (-1<alu1));
  var cast2 = bitcast<i32>((cast0<<10u));
  var cast3 = bitcast<i32>((cast1<<7u));
  var alu3 = (cast2+cast3);
  var val1 = data1_32768[(alu3+alu2)];
  var alu4 = (alu3-alu2);
  var val2 = data1_32768[(alu4+63)];
  var val3 = data1_32768[(alu4+95)];
  var gidx1 = i32(gindex.y); /* 2 */
  var alu5 = (gidx0<32);
  var alu6 = select(0.0f,val1,alu5);
  var alu7 = select(val2,0.0f,alu5);
  var alu8 = select(0.0f,val3,(alu1<0));
  var alu9 = (gidx1<1);
  var alu10 = select(0.0f,(alu6+alu7),alu9);
  var alu11 = select((alu8+val0),0.0f,alu9);
  data0_32768[(gidx0+cast2+cast3+bitcast<i32>((bitcast<u32>(gidx1)<<6u)))] = (alu10+alu11);
}`;

const E_128_2_32_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_32768:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_32768:array<f32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 32 */
  var gidx2 = i32(gindex.z); /* 128 */
  var alu0 = (gidx0+bitcast<i32>((bitcast<u32>(gidx2)<<8u)));
  var val0 = data1_32768[alu0];
  var val1 = data1_32768[(alu0+32)];
  var val2 = data1_32768[(alu0+64)];
  var val3 = data1_32768[(alu0+96)];
  var val4 = data1_32768[(alu0+128)];
  var val5 = data1_32768[(alu0+160)];
  var val6 = data1_32768[(alu0+192)];
  var val7 = data1_32768[(alu0+224)];
  var gidx1 = i32(gindex.y); /* 2 */
  var alu1 = (alu0+bitcast<i32>((bitcast<u32>(gidx1)<<5u)));
  var alu2 = -val0;
  var alu3 = -val1;
  var alu4 = -val2;
  var alu5 = -val3;
  var alu6 = -val4;
  var alu7 = -val5;
  var alu8 = -val6;
  var alu9 = -val7;
  var alu10 = select(alu2,alu3,(alu2<alu3));
  var alu11 = select(alu4,alu5,(alu4<alu5));
  var alu12 = select(alu6,alu7,(alu6<alu7));
  var alu13 = select(alu8,alu9,(alu8<alu9));
  var alu14 = (gidx1<1);
  var alu15 = select(val0,val1,(val0<val1));
  var alu16 = select(0.0f,alu15,alu14);
  var alu17 = select(-alu10,0.0f,alu14);
  var alu18 = select(val2,val3,(val2<val3));
  var alu19 = select(0.0f,alu18,alu14);
  var alu20 = select(-alu11,0.0f,alu14);
  var alu21 = select(val4,val5,(val4<val5));
  var alu22 = select(0.0f,alu21,alu14);
  var alu23 = select(-alu12,0.0f,alu14);
  var alu24 = select(val6,val7,(val6<val7));
  var alu25 = select(0.0f,alu24,alu14);
  var alu26 = select(-alu13,0.0f,alu14);
  data0_32768[alu1] = (alu16+alu17);
  data0_32768[(alu1+64)] = (alu19+alu20);
  data0_32768[(alu1+128)] = (alu22+alu23);
  data0_32768[(alu1+192)] = (alu25+alu26);
}`;

const E_16_2_2_64_8 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_32768:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_32768:array<f32>;
@compute @workgroup_size(8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 128 */
  var gidx2 = i32(gindex.z); /* 16 */
  var lidx0 = i32(lindex.x); /* 8 */
  var cast0 = bitcast<u32>(gidx2);
  var cast1 = bitcast<u32>(lidx0);
  var alu0 = (gidx0>>6u);
  var alu1 = -alu0;
  var alu2 = (gidx0&63);
  var val0 = select(0.0f, data1_32768[(bitcast<i32>((bitcast<u32>(((bitcast<i32>((cast0<<5u))+bitcast<i32>((cast1<<2u))+alu0+3)&511))<<6u))+alu2)], (-1<alu1));
  var cast2 = bitcast<i32>((cast0<<11u));
  var cast3 = bitcast<i32>((cast1<<8u));
  var alu3 = (cast2+cast3);
  var val1 = data1_32768[(alu3+alu2)];
  var alu4 = (alu3-alu2);
  var val2 = data1_32768[(alu4+127)];
  var val3 = data1_32768[(alu4+191)];
  var gidx1 = i32(gindex.y); /* 2 */
  var alu5 = (gidx0<64);
  var alu6 = select(0.0f,val1,alu5);
  var alu7 = select(val2,0.0f,alu5);
  var alu8 = select(0.0f,val3,(alu1<0));
  var alu9 = (gidx1<1);
  var alu10 = select(0.0f,(alu6+alu7),alu9);
  var alu11 = select((alu8+val0),0.0f,alu9);
  data0_32768[(gidx0+cast2+cast3+bitcast<i32>((bitcast<u32>(gidx1)<<7u)))] = (alu10+alu11);
}`;

const E_256_2_16_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_32768:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_32768:array<f32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx2 = i32(gindex.z); /* 256 */
  var alu0 = (bitcast<i32>((bitcast<u32>(gidx0)<<2u))+bitcast<i32>((bitcast<u32>(gidx2)<<7u)));
  var val0 = data1_32768[alu0];
  var val1 = data1_32768[(alu0+1)];
  var val2 = data1_32768[(alu0+2)];
  var val3 = data1_32768[(alu0+3)];
  var val4 = data1_32768[(alu0+64)];
  var val5 = data1_32768[(alu0+65)];
  var val6 = data1_32768[(alu0+66)];
  var val7 = data1_32768[(alu0+67)];
  var gidx1 = i32(gindex.y); /* 2 */
  var alu1 = (alu0+bitcast<i32>((bitcast<u32>(gidx1)<<6u)));
  var alu2 = -val0;
  var alu3 = -val1;
  var alu4 = -val2;
  var alu5 = -val3;
  var alu6 = -val4;
  var alu7 = -val5;
  var alu8 = -val6;
  var alu9 = -val7;
  var alu10 = select(alu2,alu6,(alu2<alu6));
  var alu11 = select(alu3,alu7,(alu3<alu7));
  var alu12 = select(alu4,alu8,(alu4<alu8));
  var alu13 = select(alu5,alu9,(alu5<alu9));
  var alu14 = (gidx1<1);
  var alu15 = select(val0,val4,(val0<val4));
  var alu16 = select(0.0f,alu15,alu14);
  var alu17 = select(-alu10,0.0f,alu14);
  var alu18 = select(val1,val5,(val1<val5));
  var alu19 = select(0.0f,alu18,alu14);
  var alu20 = select(-alu11,0.0f,alu14);
  var alu21 = select(val2,val6,(val2<val6));
  var alu22 = select(0.0f,alu21,alu14);
  var alu23 = select(-alu12,0.0f,alu14);
  var alu24 = select(val3,val7,(val3<val7));
  var alu25 = select(0.0f,alu24,alu14);
  var alu26 = select(-alu13,0.0f,alu14);
  data0_32768[alu1] = (alu16+alu17);
  data0_32768[(alu1+1)] = (alu19+alu20);
  data0_32768[(alu1+2)] = (alu22+alu23);
  data0_32768[(alu1+3)] = (alu25+alu26);
}`;

const E_64_2_2_32_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_32768:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_32768:array<f32>;
@compute @workgroup_size(4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 64 */
  var gidx2 = i32(gindex.z); /* 64 */
  var lidx0 = i32(lindex.x); /* 4 */
  var cast0 = bitcast<u32>(gidx2);
  var cast1 = bitcast<i32>((cast0<<9u));
  var alu0 = (gidx0&31);
  var alu1 = (lidx0+bitcast<i32>((bitcast<u32>(alu0)<<2u)));
  var val0 = data1_32768[(alu1+cast1)];
  var alu2 = (gidx0>>5u);
  var alu3 = -alu2;
  var val1 = select(0.0f, data1_32768[(alu1+bitcast<i32>((bitcast<u32>(((bitcast<i32>((cast0<<2u))+alu2+3)&255))<<7u)))], (-1<alu3));
  var alu4 = (((alu0*-4)-lidx0)+cast1);
  var val2 = data1_32768[(alu4+255)];
  var val3 = data1_32768[(alu4+383)];
  var gidx1 = i32(gindex.y); /* 2 */
  var alu5 = (gidx0<32);
  var alu6 = select(0.0f,val0,alu5);
  var alu7 = select(val2,0.0f,alu5);
  var alu8 = select(0.0f,val3,(alu3<0));
  var alu9 = (gidx1<1);
  var alu10 = select(0.0f,(alu6+alu7),alu9);
  var alu11 = select((alu8+val1),0.0f,alu9);
  data0_32768[(lidx0+bitcast<i32>((bitcast<u32>(gidx0)<<2u))+cast1+bitcast<i32>((bitcast<u32>(gidx1)<<8u)))] = (alu10+alu11);
}`;

const E_128_2_32_4n1 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_32768:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_32768:array<f32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 32 */
  var gidx2 = i32(gindex.z); /* 128 */
  var alu0 = (bitcast<i32>((bitcast<u32>(gidx0)<<2u))+bitcast<i32>((bitcast<u32>(gidx2)<<8u)));
  var val0 = data1_32768[alu0];
  var val1 = data1_32768[(alu0+1)];
  var val2 = data1_32768[(alu0+2)];
  var val3 = data1_32768[(alu0+3)];
  var val4 = data1_32768[(alu0+128)];
  var val5 = data1_32768[(alu0+129)];
  var val6 = data1_32768[(alu0+130)];
  var val7 = data1_32768[(alu0+131)];
  var gidx1 = i32(gindex.y); /* 2 */
  var alu1 = (alu0+bitcast<i32>((bitcast<u32>(gidx1)<<7u)));
  var alu2 = -val0;
  var alu3 = -val1;
  var alu4 = -val2;
  var alu5 = -val3;
  var alu6 = -val4;
  var alu7 = -val5;
  var alu8 = -val6;
  var alu9 = -val7;
  var alu10 = select(alu2,alu6,(alu2<alu6));
  var alu11 = select(alu3,alu7,(alu3<alu7));
  var alu12 = select(alu4,alu8,(alu4<alu8));
  var alu13 = select(alu5,alu9,(alu5<alu9));
  var alu14 = (gidx1<1);
  var alu15 = select(val0,val4,(val0<val4));
  var alu16 = select(0.0f,alu15,alu14);
  var alu17 = select(-alu10,0.0f,alu14);
  var alu18 = select(val1,val5,(val1<val5));
  var alu19 = select(0.0f,alu18,alu14);
  var alu20 = select(-alu11,0.0f,alu14);
  var alu21 = select(val2,val6,(val2<val6));
  var alu22 = select(0.0f,alu21,alu14);
  var alu23 = select(-alu12,0.0f,alu14);
  var alu24 = select(val3,val7,(val3<val7));
  var alu25 = select(0.0f,alu24,alu14);
  var alu26 = select(-alu13,0.0f,alu14);
  data0_32768[alu1] = (alu16+alu17);
  data0_32768[(alu1+1)] = (alu19+alu20);
  data0_32768[(alu1+2)] = (alu22+alu23);
  data0_32768[(alu1+3)] = (alu25+alu26);
}`;

const E_8_2_2_256_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_32768:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_32768:array<f32>;
@compute @workgroup_size(4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 512 */
  var gidx2 = i32(gindex.z); /* 8 */
  var lidx0 = i32(lindex.x); /* 4 */
  var cast0 = bitcast<u32>(gidx2);
  var cast1 = bitcast<u32>(lidx0);
  var alu0 = (gidx0>>8u);
  var alu1 = -alu0;
  var alu2 = (gidx0&255);
  var val0 = select(0.0f, data1_32768[(bitcast<i32>((bitcast<u32>(((bitcast<i32>((cast0<<4u))+bitcast<i32>((cast1<<2u))+alu0+3)&127))<<8u))+alu2)], (-1<alu1));
  var cast2 = bitcast<i32>((cast0<<12u));
  var cast3 = bitcast<i32>((cast1<<10u));
  var alu3 = (cast2+cast3);
  var val1 = data1_32768[(alu3+alu2)];
  var alu4 = (alu3-alu2);
  var val2 = data1_32768[(alu4+511)];
  var val3 = data1_32768[(alu4+767)];
  var gidx1 = i32(gindex.y); /* 2 */
  var alu5 = (gidx0<256);
  var alu6 = select(0.0f,val1,alu5);
  var alu7 = select(val2,0.0f,alu5);
  var alu8 = select(0.0f,val3,(alu1<0));
  var alu9 = (gidx1<1);
  var alu10 = select(0.0f,(alu6+alu7),alu9);
  var alu11 = select((alu8+val0),0.0f,alu9);
  data0_32768[(gidx0+cast2+cast3+bitcast<i32>((bitcast<u32>(gidx1)<<9u)))] = (alu10+alu11);
}`;

const E_64_2_64_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_32768:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_32768:array<f32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 64 */
  var gidx2 = i32(gindex.z); /* 64 */
  var alu0 = (bitcast<i32>((bitcast<u32>(gidx0)<<2u))+bitcast<i32>((bitcast<u32>(gidx2)<<9u)));
  var val0 = data1_32768[alu0];
  var val1 = data1_32768[(alu0+1)];
  var val2 = data1_32768[(alu0+2)];
  var val3 = data1_32768[(alu0+3)];
  var val4 = data1_32768[(alu0+256)];
  var val5 = data1_32768[(alu0+257)];
  var val6 = data1_32768[(alu0+258)];
  var val7 = data1_32768[(alu0+259)];
  var gidx1 = i32(gindex.y); /* 2 */
  var alu1 = (alu0+bitcast<i32>((bitcast<u32>(gidx1)<<8u)));
  var alu2 = -val0;
  var alu3 = -val1;
  var alu4 = -val2;
  var alu5 = -val3;
  var alu6 = -val4;
  var alu7 = -val5;
  var alu8 = -val6;
  var alu9 = -val7;
  var alu10 = select(alu2,alu6,(alu2<alu6));
  var alu11 = select(alu3,alu7,(alu3<alu7));
  var alu12 = select(alu4,alu8,(alu4<alu8));
  var alu13 = select(alu5,alu9,(alu5<alu9));
  var alu14 = (gidx1<1);
  var alu15 = select(val0,val4,(val0<val4));
  var alu16 = select(0.0f,alu15,alu14);
  var alu17 = select(-alu10,0.0f,alu14);
  var alu18 = select(val1,val5,(val1<val5));
  var alu19 = select(0.0f,alu18,alu14);
  var alu20 = select(-alu11,0.0f,alu14);
  var alu21 = select(val2,val6,(val2<val6));
  var alu22 = select(0.0f,alu21,alu14);
  var alu23 = select(-alu12,0.0f,alu14);
  var alu24 = select(val3,val7,(val3<val7));
  var alu25 = select(0.0f,alu24,alu14);
  var alu26 = select(-alu13,0.0f,alu14);
  data0_32768[alu1] = (alu16+alu17);
  data0_32768[(alu1+1)] = (alu19+alu20);
  data0_32768[(alu1+2)] = (alu22+alu23);
  data0_32768[(alu1+3)] = (alu25+alu26);
}`;

const E_2_2_512_16 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_32768:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_32768:array<f32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 512 */
  var val0 = data1_32768[gidx0];
  var val1 = data1_32768[(gidx0+2048)];
  var val2 = data1_32768[(gidx0+4096)];
  var val3 = data1_32768[(gidx0+6144)];
  var val4 = data1_32768[(gidx0+8192)];
  var val5 = data1_32768[(gidx0+10240)];
  var val6 = data1_32768[(gidx0+12288)];
  var val7 = data1_32768[(gidx0+14336)];
  var val8 = data1_32768[(gidx0+16384)];
  var val9 = data1_32768[(gidx0+18432)];
  var val10 = data1_32768[(gidx0+20480)];
  var val11 = data1_32768[(gidx0+22528)];
  var val12 = data1_32768[(gidx0+24576)];
  var val13 = data1_32768[(gidx0+26624)];
  var val14 = data1_32768[(gidx0+28672)];
  var val15 = data1_32768[(gidx0+30720)];
  var gidx1 = i32(gindex.y); /* 2 */
  var alu0 = -gidx1;
  var alu1 = (gidx0+bitcast<i32>((bitcast<u32>(gidx1)<<9u)));
  var alu2 = (-1<alu0);
  var val16 = select(0.0f, data1_32768[(alu1+1536)], alu2);
  var val17 = select(0.0f, data1_32768[(alu1+3584)], alu2);
  var val18 = select(0.0f, data1_32768[(alu1+5632)], alu2);
  var val19 = select(0.0f, data1_32768[(alu1+7680)], alu2);
  var val20 = select(0.0f, data1_32768[(alu1+9728)], alu2);
  var val21 = select(0.0f, data1_32768[(alu1+11776)], alu2);
  var val22 = select(0.0f, data1_32768[(alu1+13824)], alu2);
  var val23 = select(0.0f, data1_32768[(alu1+15872)], alu2);
  var val24 = select(0.0f, data1_32768[(alu1+17920)], alu2);
  var val25 = select(0.0f, data1_32768[(alu1+19968)], alu2);
  var val26 = select(0.0f, data1_32768[(alu1+22016)], alu2);
  var val27 = select(0.0f, data1_32768[(alu1+24064)], alu2);
  var val28 = select(0.0f, data1_32768[(alu1+26112)], alu2);
  var val29 = select(0.0f, data1_32768[(alu1+28160)], alu2);
  var val30 = select(0.0f, data1_32768[(alu1+30208)], alu2);
  var val31 = select(0.0f, data1_32768[(gidx0+(gidx1*-32256)+32256)], alu2);
  var val32 = data1_32768[(1023-gidx0)];
  var val33 = data1_32768[(1535-gidx0)];
  var val34 = data1_32768[(3071-gidx0)];
  var val35 = data1_32768[(3583-gidx0)];
  var val36 = data1_32768[(5119-gidx0)];
  var val37 = data1_32768[(5631-gidx0)];
  var val38 = data1_32768[(7167-gidx0)];
  var val39 = data1_32768[(7679-gidx0)];
  var val40 = data1_32768[(9215-gidx0)];
  var val41 = data1_32768[(9727-gidx0)];
  var val42 = data1_32768[(11263-gidx0)];
  var val43 = data1_32768[(11775-gidx0)];
  var val44 = data1_32768[(13311-gidx0)];
  var val45 = data1_32768[(13823-gidx0)];
  var val46 = data1_32768[(15359-gidx0)];
  var val47 = data1_32768[(15871-gidx0)];
  var val48 = data1_32768[(17407-gidx0)];
  var val49 = data1_32768[(17919-gidx0)];
  var val50 = data1_32768[(19455-gidx0)];
  var val51 = data1_32768[(19967-gidx0)];
  var val52 = data1_32768[(21503-gidx0)];
  var val53 = data1_32768[(22015-gidx0)];
  var val54 = data1_32768[(23551-gidx0)];
  var val55 = data1_32768[(24063-gidx0)];
  var val56 = data1_32768[(25599-gidx0)];
  var val57 = data1_32768[(26111-gidx0)];
  var val58 = data1_32768[(27647-gidx0)];
  var val59 = data1_32768[(28159-gidx0)];
  var val60 = data1_32768[(29695-gidx0)];
  var val61 = data1_32768[(30207-gidx0)];
  var val62 = data1_32768[(31743-gidx0)];
  var val63 = data1_32768[(32255-gidx0)];
  var gidx2 = i32(gindex.z); /* 2 */
  var alu3 = (alu1+bitcast<i32>((bitcast<u32>(gidx2)<<10u)));
  var alu4 = (gidx1<1);
  var alu5 = select(0.0f,val0,alu4);
  var alu6 = select(val32,0.0f,alu4);
  var alu7 = select(0.0f,val1,alu4);
  var alu8 = select(val34,0.0f,alu4);
  var alu9 = select(0.0f,val2,alu4);
  var alu10 = select(val36,0.0f,alu4);
  var alu11 = select(0.0f,val3,alu4);
  var alu12 = select(val38,0.0f,alu4);
  var alu13 = select(0.0f,val4,alu4);
  var alu14 = select(val40,0.0f,alu4);
  var alu15 = select(0.0f,val5,alu4);
  var alu16 = select(val42,0.0f,alu4);
  var alu17 = select(0.0f,val6,alu4);
  var alu18 = select(val44,0.0f,alu4);
  var alu19 = select(0.0f,val7,alu4);
  var alu20 = select(val46,0.0f,alu4);
  var alu21 = select(0.0f,val8,alu4);
  var alu22 = select(val48,0.0f,alu4);
  var alu23 = select(0.0f,val9,alu4);
  var alu24 = select(val50,0.0f,alu4);
  var alu25 = select(0.0f,val10,alu4);
  var alu26 = select(val52,0.0f,alu4);
  var alu27 = select(0.0f,val11,alu4);
  var alu28 = select(val54,0.0f,alu4);
  var alu29 = select(0.0f,val12,alu4);
  var alu30 = select(val56,0.0f,alu4);
  var alu31 = select(0.0f,val13,alu4);
  var alu32 = select(val58,0.0f,alu4);
  var alu33 = select(0.0f,val14,alu4);
  var alu34 = select(val60,0.0f,alu4);
  var alu35 = select(0.0f,val15,alu4);
  var alu36 = select(val62,0.0f,alu4);
  var alu37 = (alu0<0);
  var alu38 = select(0.0f,val33,alu37);
  var alu39 = (gidx2<1);
  var alu40 = select(0.0f,(alu5+alu6),alu39);
  var alu41 = select((alu38+val16),0.0f,alu39);
  var alu42 = select(0.0f,val35,alu37);
  var alu43 = select(0.0f,(alu7+alu8),alu39);
  var alu44 = select((alu42+val17),0.0f,alu39);
  var alu45 = select(0.0f,val37,alu37);
  var alu46 = select(0.0f,(alu9+alu10),alu39);
  var alu47 = select((alu45+val18),0.0f,alu39);
  var alu48 = select(0.0f,val39,alu37);
  var alu49 = select(0.0f,(alu11+alu12),alu39);
  var alu50 = select((alu48+val19),0.0f,alu39);
  var alu51 = select(0.0f,val41,alu37);
  var alu52 = select(0.0f,(alu13+alu14),alu39);
  var alu53 = select((alu51+val20),0.0f,alu39);
  var alu54 = select(0.0f,val43,alu37);
  var alu55 = select(0.0f,(alu15+alu16),alu39);
  var alu56 = select((alu54+val21),0.0f,alu39);
  var alu57 = select(0.0f,val45,alu37);
  var alu58 = select(0.0f,(alu17+alu18),alu39);
  var alu59 = select((alu57+val22),0.0f,alu39);
  var alu60 = select(0.0f,val47,alu37);
  var alu61 = select(0.0f,(alu19+alu20),alu39);
  var alu62 = select((alu60+val23),0.0f,alu39);
  var alu63 = select(0.0f,val49,alu37);
  var alu64 = select(0.0f,(alu21+alu22),alu39);
  var alu65 = select((alu63+val24),0.0f,alu39);
  var alu66 = select(0.0f,val51,alu37);
  var alu67 = select(0.0f,(alu23+alu24),alu39);
  var alu68 = select((alu66+val25),0.0f,alu39);
  var alu69 = select(0.0f,val53,alu37);
  var alu70 = select(0.0f,(alu25+alu26),alu39);
  var alu71 = select((alu69+val26),0.0f,alu39);
  var alu72 = select(0.0f,val55,alu37);
  var alu73 = select(0.0f,(alu27+alu28),alu39);
  var alu74 = select((alu72+val27),0.0f,alu39);
  var alu75 = select(0.0f,val57,alu37);
  var alu76 = select(0.0f,(alu29+alu30),alu39);
  var alu77 = select((alu75+val28),0.0f,alu39);
  var alu78 = select(0.0f,val59,alu37);
  var alu79 = select(0.0f,(alu31+alu32),alu39);
  var alu80 = select((alu78+val29),0.0f,alu39);
  var alu81 = select(0.0f,val61,alu37);
  var alu82 = select(0.0f,(alu33+alu34),alu39);
  var alu83 = select((alu81+val30),0.0f,alu39);
  var alu84 = select(0.0f,val63,alu37);
  var alu85 = select(0.0f,(alu35+alu36),alu39);
  var alu86 = select((alu84+val31),0.0f,alu39);
  data0_32768[alu3] = (alu40+alu41);
  data0_32768[(alu3+2048)] = (alu43+alu44);
  data0_32768[(alu3+4096)] = (alu46+alu47);
  data0_32768[(alu3+6144)] = (alu49+alu50);
  data0_32768[(alu3+8192)] = (alu52+alu53);
  data0_32768[(alu3+10240)] = (alu55+alu56);
  data0_32768[(alu3+12288)] = (alu58+alu59);
  data0_32768[(alu3+14336)] = (alu61+alu62);
  data0_32768[(alu3+16384)] = (alu64+alu65);
  data0_32768[(alu3+18432)] = (alu67+alu68);
  data0_32768[(alu3+20480)] = (alu70+alu71);
  data0_32768[(alu3+22528)] = (alu73+alu74);
  data0_32768[(alu3+24576)] = (alu76+alu77);
  data0_32768[(alu3+26624)] = (alu79+alu80);
  data0_32768[(alu3+28672)] = (alu82+alu83);
  data0_32768[(alu3+30720)] = (alu85+alu86);
}`;

const E_8_2_512_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_32768:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_32768:array<f32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 512 */
  var gidx2 = i32(gindex.z); /* 8 */
  var alu0 = (gidx0+bitcast<i32>((bitcast<u32>(gidx2)<<12u)));
  var val0 = data1_32768[alu0];
  var val1 = data1_32768[(alu0+512)];
  var val2 = data1_32768[(alu0+1024)];
  var val3 = data1_32768[(alu0+1536)];
  var val4 = data1_32768[(alu0+2048)];
  var val5 = data1_32768[(alu0+2560)];
  var val6 = data1_32768[(alu0+3072)];
  var val7 = data1_32768[(alu0+3584)];
  var gidx1 = i32(gindex.y); /* 2 */
  var alu1 = (alu0+bitcast<i32>((bitcast<u32>(gidx1)<<9u)));
  var alu2 = -val0;
  var alu3 = -val1;
  var alu4 = -val2;
  var alu5 = -val3;
  var alu6 = -val4;
  var alu7 = -val5;
  var alu8 = -val6;
  var alu9 = -val7;
  var alu10 = select(alu2,alu3,(alu2<alu3));
  var alu11 = select(alu4,alu5,(alu4<alu5));
  var alu12 = select(alu6,alu7,(alu6<alu7));
  var alu13 = select(alu8,alu9,(alu8<alu9));
  var alu14 = (gidx1<1);
  var alu15 = select(val0,val1,(val0<val1));
  var alu16 = select(0.0f,alu15,alu14);
  var alu17 = select(-alu10,0.0f,alu14);
  var alu18 = select(val2,val3,(val2<val3));
  var alu19 = select(0.0f,alu18,alu14);
  var alu20 = select(-alu11,0.0f,alu14);
  var alu21 = select(val4,val5,(val4<val5));
  var alu22 = select(0.0f,alu21,alu14);
  var alu23 = select(-alu12,0.0f,alu14);
  var alu24 = select(val6,val7,(val6<val7));
  var alu25 = select(0.0f,alu24,alu14);
  var alu26 = select(-alu13,0.0f,alu14);
  data0_32768[alu1] = (alu16+alu17);
  data0_32768[(alu1+1024)] = (alu19+alu20);
  data0_32768[(alu1+2048)] = (alu22+alu23);
  data0_32768[(alu1+3072)] = (alu25+alu26);
}`;

const E_2_2_2_1024_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_32768:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_32768:array<f32>;
@compute @workgroup_size(4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 2048 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 4 */
  var cast0 = bitcast<u32>(gidx2);
  var cast1 = bitcast<u32>(lidx0);
  var alu0 = (gidx0>>10u);
  var alu1 = -alu0;
  var alu2 = (gidx0&1023);
  var val0 = select(0.0f, data1_32768[(bitcast<i32>((bitcast<u32>(((bitcast<i32>((cast0<<4u))+bitcast<i32>((cast1<<2u))+alu0+3)&31))<<10u))+alu2)], (-1<alu1));
  var cast2 = bitcast<i32>((cast0<<14u));
  var cast3 = bitcast<i32>((cast1<<12u));
  var alu3 = (cast2+cast3);
  var val1 = data1_32768[(alu3+alu2)];
  var alu4 = (alu3-alu2);
  var val2 = data1_32768[(alu4+2047)];
  var val3 = data1_32768[(alu4+3071)];
  var gidx1 = i32(gindex.y); /* 2 */
  var alu5 = (gidx0<1024);
  var alu6 = select(0.0f,val1,alu5);
  var alu7 = select(val2,0.0f,alu5);
  var alu8 = select(0.0f,val3,(alu1<0));
  var alu9 = (gidx1<1);
  var alu10 = select(0.0f,(alu6+alu7),alu9);
  var alu11 = select((alu8+val0),0.0f,alu9);
  data0_32768[(gidx0+cast2+cast3+bitcast<i32>((bitcast<u32>(gidx1)<<11u)))] = (alu10+alu11);
}`;

const E_2_1024_16 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_32768:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_32768:array<f32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 1024 */
  var val0 = data1_32768[gidx0];
  var val1 = data1_32768[(gidx0+1024)];
  var val2 = data1_32768[(gidx0+2048)];
  var val3 = data1_32768[(gidx0+3072)];
  var val4 = data1_32768[(gidx0+4096)];
  var val5 = data1_32768[(gidx0+5120)];
  var val6 = data1_32768[(gidx0+6144)];
  var val7 = data1_32768[(gidx0+7168)];
  var val8 = data1_32768[(gidx0+8192)];
  var val9 = data1_32768[(gidx0+9216)];
  var val10 = data1_32768[(gidx0+10240)];
  var val11 = data1_32768[(gidx0+11264)];
  var val12 = data1_32768[(gidx0+12288)];
  var val13 = data1_32768[(gidx0+13312)];
  var val14 = data1_32768[(gidx0+14336)];
  var val15 = data1_32768[(gidx0+15360)];
  var val16 = data1_32768[(gidx0+16384)];
  var val17 = data1_32768[(gidx0+17408)];
  var val18 = data1_32768[(gidx0+18432)];
  var val19 = data1_32768[(gidx0+19456)];
  var val20 = data1_32768[(gidx0+20480)];
  var val21 = data1_32768[(gidx0+21504)];
  var val22 = data1_32768[(gidx0+22528)];
  var val23 = data1_32768[(gidx0+23552)];
  var val24 = data1_32768[(gidx0+24576)];
  var val25 = data1_32768[(gidx0+25600)];
  var val26 = data1_32768[(gidx0+26624)];
  var val27 = data1_32768[(gidx0+27648)];
  var val28 = data1_32768[(gidx0+28672)];
  var val29 = data1_32768[(gidx0+29696)];
  var val30 = data1_32768[(gidx0+30720)];
  var val31 = data1_32768[(gidx0+31744)];
  var gidx1 = i32(gindex.y); /* 2 */
  var alu0 = (gidx0+bitcast<i32>((bitcast<u32>(gidx1)<<10u)));
  var alu1 = -val0;
  var alu2 = -val1;
  var alu3 = -val2;
  var alu4 = -val3;
  var alu5 = -val4;
  var alu6 = -val5;
  var alu7 = -val6;
  var alu8 = -val7;
  var alu9 = -val8;
  var alu10 = -val9;
  var alu11 = -val10;
  var alu12 = -val11;
  var alu13 = -val12;
  var alu14 = -val13;
  var alu15 = -val14;
  var alu16 = -val15;
  var alu17 = -val16;
  var alu18 = -val17;
  var alu19 = -val18;
  var alu20 = -val19;
  var alu21 = -val20;
  var alu22 = -val21;
  var alu23 = -val22;
  var alu24 = -val23;
  var alu25 = -val24;
  var alu26 = -val25;
  var alu27 = -val26;
  var alu28 = -val27;
  var alu29 = -val28;
  var alu30 = -val29;
  var alu31 = -val30;
  var alu32 = -val31;
  var alu33 = select(alu1,alu2,(alu1<alu2));
  var alu34 = select(alu3,alu4,(alu3<alu4));
  var alu35 = select(alu5,alu6,(alu5<alu6));
  var alu36 = select(alu7,alu8,(alu7<alu8));
  var alu37 = select(alu9,alu10,(alu9<alu10));
  var alu38 = select(alu11,alu12,(alu11<alu12));
  var alu39 = select(alu13,alu14,(alu13<alu14));
  var alu40 = select(alu15,alu16,(alu15<alu16));
  var alu41 = select(alu17,alu18,(alu17<alu18));
  var alu42 = select(alu19,alu20,(alu19<alu20));
  var alu43 = select(alu21,alu22,(alu21<alu22));
  var alu44 = select(alu23,alu24,(alu23<alu24));
  var alu45 = select(alu25,alu26,(alu25<alu26));
  var alu46 = select(alu27,alu28,(alu27<alu28));
  var alu47 = select(alu29,alu30,(alu29<alu30));
  var alu48 = select(alu31,alu32,(alu31<alu32));
  var alu49 = (gidx1<1);
  var alu50 = select(val0,val1,(val0<val1));
  var alu51 = select(0.0f,alu50,alu49);
  var alu52 = select(-alu33,0.0f,alu49);
  var alu53 = select(val2,val3,(val2<val3));
  var alu54 = select(0.0f,alu53,alu49);
  var alu55 = select(-alu34,0.0f,alu49);
  var alu56 = select(val4,val5,(val4<val5));
  var alu57 = select(0.0f,alu56,alu49);
  var alu58 = select(-alu35,0.0f,alu49);
  var alu59 = select(val6,val7,(val6<val7));
  var alu60 = select(0.0f,alu59,alu49);
  var alu61 = select(-alu36,0.0f,alu49);
  var alu62 = select(val8,val9,(val8<val9));
  var alu63 = select(0.0f,alu62,alu49);
  var alu64 = select(-alu37,0.0f,alu49);
  var alu65 = select(val10,val11,(val10<val11));
  var alu66 = select(0.0f,alu65,alu49);
  var alu67 = select(-alu38,0.0f,alu49);
  var alu68 = select(val12,val13,(val12<val13));
  var alu69 = select(0.0f,alu68,alu49);
  var alu70 = select(-alu39,0.0f,alu49);
  var alu71 = select(val14,val15,(val14<val15));
  var alu72 = select(0.0f,alu71,alu49);
  var alu73 = select(-alu40,0.0f,alu49);
  var alu74 = select(val16,val17,(val16<val17));
  var alu75 = select(0.0f,alu74,alu49);
  var alu76 = select(-alu41,0.0f,alu49);
  var alu77 = select(val18,val19,(val18<val19));
  var alu78 = select(0.0f,alu77,alu49);
  var alu79 = select(-alu42,0.0f,alu49);
  var alu80 = select(val20,val21,(val20<val21));
  var alu81 = select(0.0f,alu80,alu49);
  var alu82 = select(-alu43,0.0f,alu49);
  var alu83 = select(val22,val23,(val22<val23));
  var alu84 = select(0.0f,alu83,alu49);
  var alu85 = select(-alu44,0.0f,alu49);
  var alu86 = select(val24,val25,(val24<val25));
  var alu87 = select(0.0f,alu86,alu49);
  var alu88 = select(-alu45,0.0f,alu49);
  var alu89 = select(val26,val27,(val26<val27));
  var alu90 = select(0.0f,alu89,alu49);
  var alu91 = select(-alu46,0.0f,alu49);
  var alu92 = select(val28,val29,(val28<val29));
  var alu93 = select(0.0f,alu92,alu49);
  var alu94 = select(-alu47,0.0f,alu49);
  var alu95 = select(val30,val31,(val30<val31));
  var alu96 = select(0.0f,alu95,alu49);
  var alu97 = select(-alu48,0.0f,alu49);
  data0_32768[alu0] = (alu51+alu52);
  data0_32768[(alu0+2048)] = (alu54+alu55);
  data0_32768[(alu0+4096)] = (alu57+alu58);
  data0_32768[(alu0+6144)] = (alu60+alu61);
  data0_32768[(alu0+8192)] = (alu63+alu64);
  data0_32768[(alu0+10240)] = (alu66+alu67);
  data0_32768[(alu0+12288)] = (alu69+alu70);
  data0_32768[(alu0+14336)] = (alu72+alu73);
  data0_32768[(alu0+16384)] = (alu75+alu76);
  data0_32768[(alu0+18432)] = (alu78+alu79);
  data0_32768[(alu0+20480)] = (alu81+alu82);
  data0_32768[(alu0+22528)] = (alu84+alu85);
  data0_32768[(alu0+24576)] = (alu87+alu88);
  data0_32768[(alu0+26624)] = (alu90+alu91);
  data0_32768[(alu0+28672)] = (alu93+alu94);
  data0_32768[(alu0+30720)] = (alu96+alu97);
}`;

const E_4_2_2048_2 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_32768:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_32768:array<f32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 2048 */
  var gidx2 = i32(gindex.z); /* 4 */
  var cast0 = bitcast<u32>(gidx2);
  var cast1 = bitcast<i32>((cast0<<13u));
  var alu0 = (gidx0+cast1);
  var val0 = data1_32768[alu0];
  var gidx1 = i32(gindex.y); /* 2 */
  var alu1 = -gidx1;
  var val1 = select(0.0f, data1_32768[(gidx0+bitcast<i32>((bitcast<u32>(((gidx1+bitcast<i32>((cast0<<2u))+3)&15))<<11u)))], (-1<alu1));
  var alu2 = (cast1-gidx0);
  var val2 = data1_32768[(alu2+4095)];
  var val3 = data1_32768[(alu2+6143)];
  var alu3 = (alu0+bitcast<i32>((bitcast<u32>(gidx1)<<11u)));
  var alu4 = (gidx1<1);
  var alu5 = select(0.0f,val0,alu4);
  var alu6 = select(val2,0.0f,alu4);
  var alu7 = select(0.0f,val3,(alu1<0));
  data0_32768[alu3] = (alu5+alu6);
  data0_32768[(alu3+4096)] = (alu7+val1);
}`;

const E_2_2048_8 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_32768:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_32768:array<f32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 2048 */
  var val0 = data1_32768[gidx0];
  var val1 = data1_32768[(gidx0+2048)];
  var val2 = data1_32768[(gidx0+4096)];
  var val3 = data1_32768[(gidx0+6144)];
  var val4 = data1_32768[(gidx0+8192)];
  var val5 = data1_32768[(gidx0+10240)];
  var val6 = data1_32768[(gidx0+12288)];
  var val7 = data1_32768[(gidx0+14336)];
  var val8 = data1_32768[(gidx0+16384)];
  var val9 = data1_32768[(gidx0+18432)];
  var val10 = data1_32768[(gidx0+20480)];
  var val11 = data1_32768[(gidx0+22528)];
  var val12 = data1_32768[(gidx0+24576)];
  var val13 = data1_32768[(gidx0+26624)];
  var val14 = data1_32768[(gidx0+28672)];
  var val15 = data1_32768[(gidx0+30720)];
  var gidx1 = i32(gindex.y); /* 2 */
  var alu0 = (gidx0+bitcast<i32>((bitcast<u32>(gidx1)<<11u)));
  var alu1 = -val0;
  var alu2 = -val1;
  var alu3 = -val2;
  var alu4 = -val3;
  var alu5 = -val4;
  var alu6 = -val5;
  var alu7 = -val6;
  var alu8 = -val7;
  var alu9 = -val8;
  var alu10 = -val9;
  var alu11 = -val10;
  var alu12 = -val11;
  var alu13 = -val12;
  var alu14 = -val13;
  var alu15 = -val14;
  var alu16 = -val15;
  var alu17 = select(alu1,alu2,(alu1<alu2));
  var alu18 = select(alu3,alu4,(alu3<alu4));
  var alu19 = select(alu5,alu6,(alu5<alu6));
  var alu20 = select(alu7,alu8,(alu7<alu8));
  var alu21 = select(alu9,alu10,(alu9<alu10));
  var alu22 = select(alu11,alu12,(alu11<alu12));
  var alu23 = select(alu13,alu14,(alu13<alu14));
  var alu24 = select(alu15,alu16,(alu15<alu16));
  var alu25 = (gidx1<1);
  var alu26 = select(val0,val1,(val0<val1));
  var alu27 = select(0.0f,alu26,alu25);
  var alu28 = select(-alu17,0.0f,alu25);
  var alu29 = select(val2,val3,(val2<val3));
  var alu30 = select(0.0f,alu29,alu25);
  var alu31 = select(-alu18,0.0f,alu25);
  var alu32 = select(val4,val5,(val4<val5));
  var alu33 = select(0.0f,alu32,alu25);
  var alu34 = select(-alu19,0.0f,alu25);
  var alu35 = select(val6,val7,(val6<val7));
  var alu36 = select(0.0f,alu35,alu25);
  var alu37 = select(-alu20,0.0f,alu25);
  var alu38 = select(val8,val9,(val8<val9));
  var alu39 = select(0.0f,alu38,alu25);
  var alu40 = select(-alu21,0.0f,alu25);
  var alu41 = select(val10,val11,(val10<val11));
  var alu42 = select(0.0f,alu41,alu25);
  var alu43 = select(-alu22,0.0f,alu25);
  var alu44 = select(val12,val13,(val12<val13));
  var alu45 = select(0.0f,alu44,alu25);
  var alu46 = select(-alu23,0.0f,alu25);
  var alu47 = select(val14,val15,(val14<val15));
  var alu48 = select(0.0f,alu47,alu25);
  var alu49 = select(-alu24,0.0f,alu25);
  data0_32768[alu0] = (alu27+alu28);
  data0_32768[(alu0+4096)] = (alu30+alu31);
  data0_32768[(alu0+8192)] = (alu33+alu34);
  data0_32768[(alu0+12288)] = (alu36+alu37);
  data0_32768[(alu0+16384)] = (alu39+alu40);
  data0_32768[(alu0+20480)] = (alu42+alu43);
  data0_32768[(alu0+24576)] = (alu45+alu46);
  data0_32768[(alu0+28672)] = (alu48+alu49);
}`;

const E_2_2_2_1024_4n1 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_32768:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_32768:array<f32>;
@compute @workgroup_size(4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 2048 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 4 */
  var cast0 = bitcast<u32>(gidx2);
  var cast1 = bitcast<i32>((cast0<<14u));
  var alu0 = (gidx0&1023);
  var alu1 = (lidx0+bitcast<i32>((bitcast<u32>(alu0)<<2u)));
  var val0 = data1_32768[(alu1+cast1)];
  var alu2 = (gidx0>>10u);
  var alu3 = -alu2;
  var val1 = select(0.0f, data1_32768[(alu1+bitcast<i32>((bitcast<u32>(((bitcast<i32>((cast0<<2u))+alu2+3)&7))<<12u)))], (-1<alu3));
  var alu4 = (((alu0*-4)-lidx0)+cast1);
  var val2 = data1_32768[(alu4+8191)];
  var val3 = data1_32768[(alu4+12287)];
  var gidx1 = i32(gindex.y); /* 2 */
  var alu5 = (gidx0<1024);
  var alu6 = select(0.0f,val0,alu5);
  var alu7 = select(val2,0.0f,alu5);
  var alu8 = select(0.0f,val3,(alu3<0));
  var alu9 = (gidx1<1);
  var alu10 = select(0.0f,(alu6+alu7),alu9);
  var alu11 = select((alu8+val1),0.0f,alu9);
  data0_32768[(lidx0+bitcast<i32>((bitcast<u32>(gidx0)<<2u))+bitcast<i32>((bitcast<u32>(gidx1)<<13u))+cast1)] = (alu10+alu11);
}`;

const E_2_4096_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_32768:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_32768:array<f32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 4096 */
  var val0 = data1_32768[gidx0];
  var val1 = data1_32768[(gidx0+4096)];
  var val2 = data1_32768[(gidx0+8192)];
  var val3 = data1_32768[(gidx0+12288)];
  var val4 = data1_32768[(gidx0+16384)];
  var val5 = data1_32768[(gidx0+20480)];
  var val6 = data1_32768[(gidx0+24576)];
  var val7 = data1_32768[(gidx0+28672)];
  var gidx1 = i32(gindex.y); /* 2 */
  var alu0 = (gidx0+bitcast<i32>((bitcast<u32>(gidx1)<<12u)));
  var alu1 = -val0;
  var alu2 = -val1;
  var alu3 = -val2;
  var alu4 = -val3;
  var alu5 = -val4;
  var alu6 = -val5;
  var alu7 = -val6;
  var alu8 = -val7;
  var alu9 = select(alu1,alu2,(alu1<alu2));
  var alu10 = select(alu3,alu4,(alu3<alu4));
  var alu11 = select(alu5,alu6,(alu5<alu6));
  var alu12 = select(alu7,alu8,(alu7<alu8));
  var alu13 = (gidx1<1);
  var alu14 = select(val0,val1,(val0<val1));
  var alu15 = select(0.0f,alu14,alu13);
  var alu16 = select(-alu9,0.0f,alu13);
  var alu17 = select(val2,val3,(val2<val3));
  var alu18 = select(0.0f,alu17,alu13);
  var alu19 = select(-alu10,0.0f,alu13);
  var alu20 = select(val4,val5,(val4<val5));
  var alu21 = select(0.0f,alu20,alu13);
  var alu22 = select(-alu11,0.0f,alu13);
  var alu23 = select(val6,val7,(val6<val7));
  var alu24 = select(0.0f,alu23,alu13);
  var alu25 = select(-alu12,0.0f,alu13);
  data0_32768[alu0] = (alu15+alu16);
  data0_32768[(alu0+8192)] = (alu18+alu19);
  data0_32768[(alu0+16384)] = (alu21+alu22);
  data0_32768[(alu0+24576)] = (alu24+alu25);
}`;

const E_2_2_2048_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_32768:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_32768:array<f32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 2048 */
  var cast0 = bitcast<i32>((bitcast<u32>(gidx0)<<2u));
  var val0 = data1_32768[cast0];
  var val1 = data1_32768[(cast0+1)];
  var val2 = data1_32768[(cast0+2)];
  var val3 = data1_32768[(cast0+3)];
  var gidx1 = i32(gindex.y); /* 2 */
  var alu0 = -gidx1;
  var alu1 = (cast0+(gidx1*-24576));
  var alu2 = (-1<alu0);
  var val4 = select(0.0f, data1_32768[(alu1+24576)], alu2);
  var val5 = select(0.0f, data1_32768[(alu1+24577)], alu2);
  var val6 = select(0.0f, data1_32768[(alu1+24578)], alu2);
  var val7 = select(0.0f, data1_32768[(alu1+24579)], alu2);
  var alu3 = (gidx0*-4);
  var val8 = data1_32768[(alu3+16380)];
  var val9 = data1_32768[(alu3+16381)];
  var val10 = data1_32768[(alu3+16382)];
  var val11 = data1_32768[(alu3+16383)];
  var val12 = data1_32768[(alu3+24572)];
  var val13 = data1_32768[(alu3+24573)];
  var val14 = data1_32768[(alu3+24574)];
  var val15 = data1_32768[(alu3+24575)];
  var gidx2 = i32(gindex.z); /* 2 */
  var alu4 = (cast0+bitcast<i32>((bitcast<u32>(gidx1)<<13u))+bitcast<i32>((bitcast<u32>(gidx2)<<14u)));
  var alu5 = (gidx1<1);
  var alu6 = select(0.0f,val0,alu5);
  var alu7 = select(val11,0.0f,alu5);
  var alu8 = select(0.0f,val1,alu5);
  var alu9 = select(val10,0.0f,alu5);
  var alu10 = select(0.0f,val2,alu5);
  var alu11 = select(val9,0.0f,alu5);
  var alu12 = select(0.0f,val3,alu5);
  var alu13 = select(val8,0.0f,alu5);
  var alu14 = (alu0<0);
  var alu15 = select(0.0f,val15,alu14);
  var alu16 = (gidx2<1);
  var alu17 = select(0.0f,(alu6+alu7),alu16);
  var alu18 = select((alu15+val4),0.0f,alu16);
  var alu19 = select(0.0f,val14,alu14);
  var alu20 = select(0.0f,(alu8+alu9),alu16);
  var alu21 = select((alu19+val5),0.0f,alu16);
  var alu22 = select(0.0f,val13,alu14);
  var alu23 = select(0.0f,(alu10+alu11),alu16);
  var alu24 = select((alu22+val6),0.0f,alu16);
  var alu25 = select(0.0f,val12,alu14);
  var alu26 = select(0.0f,(alu12+alu13),alu16);
  var alu27 = select((alu25+val7),0.0f,alu16);
  data0_32768[alu4] = (alu17+alu18);
  data0_32768[(alu4+1)] = (alu20+alu21);
  data0_32768[(alu4+2)] = (alu23+alu24);
  data0_32768[(alu4+3)] = (alu26+alu27);
}`;

const E_2_2_2048_4n1 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_32768:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_32768:array<f32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 2048 */
  var gidx2 = i32(gindex.z); /* 2 */
  var cast0 = bitcast<i32>((bitcast<u32>(gidx0)<<2u));
  var cast1 = bitcast<i32>((bitcast<u32>(gidx2)<<14u));
  var alu0 = (cast0+cast1);
  var val0 = data1_32768[alu0];
  var val1 = data1_32768[(alu0+1)];
  var val2 = data1_32768[(alu0+2)];
  var val3 = data1_32768[(alu0+3)];
  var val4 = data1_32768[(alu0+8192)];
  var val5 = data1_32768[(alu0+8193)];
  var val6 = data1_32768[(alu0+8194)];
  var val7 = data1_32768[(alu0+8195)];
  var gidx1 = i32(gindex.y); /* 2 */
  var alu1 = (cast0+bitcast<i32>((bitcast<u32>(gidx1)<<13u))+cast1);
  var alu2 = -val0;
  var alu3 = -val1;
  var alu4 = -val2;
  var alu5 = -val3;
  var alu6 = -val4;
  var alu7 = -val5;
  var alu8 = -val6;
  var alu9 = -val7;
  var alu10 = select(alu2,alu6,(alu2<alu6));
  var alu11 = select(alu3,alu7,(alu3<alu7));
  var alu12 = select(alu4,alu8,(alu4<alu8));
  var alu13 = select(alu5,alu9,(alu5<alu9));
  var alu14 = (gidx1<1);
  var alu15 = select(val0,val4,(val0<val4));
  var alu16 = select(0.0f,alu15,alu14);
  var alu17 = select(-alu10,0.0f,alu14);
  var alu18 = select(val1,val5,(val1<val5));
  var alu19 = select(0.0f,alu18,alu14);
  var alu20 = select(-alu11,0.0f,alu14);
  var alu21 = select(val2,val6,(val2<val6));
  var alu22 = select(0.0f,alu21,alu14);
  var alu23 = select(-alu12,0.0f,alu14);
  var alu24 = select(val3,val7,(val3<val7));
  var alu25 = select(0.0f,alu24,alu14);
  var alu26 = select(-alu13,0.0f,alu14);
  data0_32768[alu1] = (alu16+alu17);
  data0_32768[(alu1+1)] = (alu19+alu20);
  data0_32768[(alu1+2)] = (alu22+alu23);
  data0_32768[(alu1+3)] = (alu25+alu26);
}`;

const E_2_4096_4n1 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_32768:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_32768:array<f32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 4096 */
  var cast0 = bitcast<i32>((bitcast<u32>(gidx0)<<2u));
  var val0 = data1_32768[cast0];
  var val1 = data1_32768[(cast0+1)];
  var val2 = data1_32768[(cast0+2)];
  var val3 = data1_32768[(cast0+3)];
  var alu0 = (gidx0*-4);
  var val4 = data1_32768[(alu0+32764)];
  var val5 = data1_32768[(alu0+32765)];
  var val6 = data1_32768[(alu0+32766)];
  var val7 = data1_32768[(alu0+32767)];
  var gidx1 = i32(gindex.y); /* 2 */
  var alu1 = (cast0+bitcast<i32>((bitcast<u32>(gidx1)<<14u)));
  var alu2 = -val0;
  var alu3 = -val1;
  var alu4 = -val2;
  var alu5 = -val3;
  var alu6 = -val4;
  var alu7 = -val5;
  var alu8 = -val6;
  var alu9 = -val7;
  var alu10 = select(alu2,alu9,(alu2<alu9));
  var alu11 = select(alu3,alu8,(alu3<alu8));
  var alu12 = select(alu4,alu7,(alu4<alu7));
  var alu13 = select(alu5,alu6,(alu5<alu6));
  var alu14 = (gidx1<1);
  var alu15 = select(val0,val7,(val0<val7));
  var alu16 = select(0.0f,alu15,alu14);
  var alu17 = select(-alu10,0.0f,alu14);
  var alu18 = select(val1,val6,(val1<val6));
  var alu19 = select(0.0f,alu18,alu14);
  var alu20 = select(-alu11,0.0f,alu14);
  var alu21 = select(val2,val5,(val2<val5));
  var alu22 = select(0.0f,alu21,alu14);
  var alu23 = select(-alu12,0.0f,alu14);
  var alu24 = select(val3,val4,(val3<val4));
  var alu25 = select(0.0f,alu24,alu14);
  var alu26 = select(-alu13,0.0f,alu14);
  data0_32768[alu1] = (alu16+alu17);
  data0_32768[(alu1+1)] = (alu19+alu20);
  data0_32768[(alu1+2)] = (alu22+alu23);
  data0_32768[(alu1+3)] = (alu25+alu26);
}`;

const r_1950_28_7_2_975n1 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<i32,392>;
@group(0) @binding(1)var<storage,read_write>data0_27300:array<i32>;
@group(0) @binding(2)var<storage,read_write>data1_32768:array<f32>;
@compute @workgroup_size(28) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<i32,14>;
  var acc1: array<i32,14>;
  var gidx0 = i32(gindex.x); /* 1950 */
  var alu0 = (gidx0*14);
  var alu1 = (alu0+1);
  var val0 = data1_32768[alu1];
  var alu2 = (alu0+2);
  var val1 = data1_32768[alu2];
  var alu3 = (alu0+3);
  var val2 = data1_32768[alu3];
  var alu4 = (alu0+4);
  var val3 = data1_32768[alu4];
  var alu5 = (alu0+5);
  var val4 = data1_32768[alu5];
  var alu6 = (alu0+6);
  var val5 = data1_32768[alu6];
  var alu7 = (alu0+7);
  var val6 = data1_32768[alu7];
  var alu8 = (alu0+8);
  var val7 = data1_32768[alu8];
  var alu9 = (alu0+9);
  var val8 = data1_32768[alu9];
  var alu10 = (alu0+10);
  var val9 = data1_32768[alu10];
  var alu11 = (alu0+11);
  var val10 = data1_32768[alu11];
  var alu12 = (alu0+12);
  var val11 = data1_32768[alu12];
  var alu13 = (alu0+13);
  var val12 = data1_32768[alu13];
  var val13 = data1_32768[alu0];
  var lidx0 = i32(lindex.x); /* 28 */
  acc0[0] = 0;
  acc0[1] = 0;
  acc0[2] = 0;
  acc0[3] = 0;
  acc0[4] = 0;
  acc0[5] = 0;
  acc0[6] = 0;
  acc0[7] = 0;
  acc0[8] = 0;
  acc0[9] = 0;
  acc0[10] = 0;
  acc0[11] = 0;
  acc0[12] = 0;
  acc0[13] = 0;
  for (var Ridx0 = 0; Ridx0 < 975; Ridx0++) {
    var alu28 = ((lidx0*975)+Ridx0);
    var val14 = data1_32768[alu28];
    acc0[0] = (acc0[0]+(i32(((alu28<alu1)&(val14==val13)))));
    acc0[1] = (acc0[1]+(i32(((alu28<alu8)&(val14==val6)))));
    acc0[2] = (acc0[2]+(i32(((alu28<alu2)&(val14==val0)))));
    acc0[3] = (acc0[3]+(i32(((alu28<alu9)&(val14==val7)))));
    acc0[4] = (acc0[4]+(i32(((alu28<alu3)&(val14==val1)))));
    acc0[5] = (acc0[5]+(i32(((alu28<alu10)&(val14==val8)))));
    acc0[6] = (acc0[6]+(i32(((alu28<alu4)&(val14==val2)))));
    acc0[7] = (acc0[7]+(i32(((alu28<alu11)&(val14==val9)))));
    acc0[8] = (acc0[8]+(i32(((alu28<alu5)&(val14==val3)))));
    acc0[9] = (acc0[9]+(i32(((alu28<alu12)&(val14==val10)))));
    acc0[10] = (acc0[10]+(i32(((alu28<alu6)&(val14==val4)))));
    acc0[11] = (acc0[11]+(i32(((alu28<alu13)&(val14==val11)))));
    acc0[12] = (acc0[12]+(i32(((alu28<alu7)&(val14==val5)))));
    acc0[13] = (acc0[13]+(i32(((alu28<(alu0+14))&(val14==val12)))));
  }
  var alu44 = (lidx0*14);
  temp0[(alu44+1)] = acc0[1];
  temp0[(alu44+2)] = acc0[2];
  temp0[(alu44+3)] = acc0[3];
  temp0[(alu44+4)] = acc0[4];
  temp0[(alu44+5)] = acc0[5];
  temp0[(alu44+6)] = acc0[6];
  temp0[(alu44+7)] = acc0[7];
  temp0[(alu44+8)] = acc0[8];
  temp0[(alu44+9)] = acc0[9];
  temp0[(alu44+10)] = acc0[10];
  temp0[(alu44+11)] = acc0[11];
  temp0[(alu44+12)] = acc0[12];
  temp0[(alu44+13)] = acc0[13];
  temp0[alu44] = acc0[0];
  workgroupBarrier();
  acc1[0] = 0;
  acc1[1] = 0;
  acc1[2] = 0;
  acc1[3] = 0;
  acc1[4] = 0;
  acc1[5] = 0;
  acc1[6] = 0;
  acc1[7] = 0;
  acc1[8] = 0;
  acc1[9] = 0;
  acc1[10] = 0;
  acc1[11] = 0;
  acc1[12] = 0;
  acc1[13] = 0;
  for (var Ridx102 = 0; Ridx102 < 28; Ridx102++) {
    var alu74 = (Ridx102*14);
    var val15 = temp0[(alu74+5)];
    var val16 = temp0[alu74];
    var val17 = temp0[(alu74+1)];
    var val18 = temp0[(alu74+2)];
    var val19 = temp0[(alu74+3)];
    var val20 = temp0[(alu74+4)];
    var val21 = temp0[(alu74+6)];
    var val22 = temp0[(alu74+7)];
    var val23 = temp0[(alu74+8)];
    var val24 = temp0[(alu74+9)];
    var val25 = temp0[(alu74+10)];
    var val26 = temp0[(alu74+11)];
    var val27 = temp0[(alu74+12)];
    var val28 = temp0[(alu74+13)];
    acc1[0] = (acc1[0]+val16);
    acc1[1] = (acc1[1]+val17);
    acc1[2] = (acc1[2]+val18);
    acc1[3] = (acc1[3]+val19);
    acc1[4] = (acc1[4]+val20);
    acc1[5] = (acc1[5]+val15);
    acc1[6] = (acc1[6]+val21);
    acc1[7] = (acc1[7]+val22);
    acc1[8] = (acc1[8]+val23);
    acc1[9] = (acc1[9]+val24);
    acc1[10] = (acc1[10]+val25);
    acc1[11] = (acc1[11]+val26);
    acc1[12] = (acc1[12]+val27);
    acc1[13] = (acc1[13]+val28);
  }
  var alu90 = (lidx0==0);
  if (alu90) {
    data0_27300[alu1] = acc1[2];
  }
  if (alu90) {
    data0_27300[alu2] = acc1[4];
  }
  if (alu90) {
    data0_27300[alu3] = acc1[6];
  }
  if (alu90) {
    data0_27300[alu4] = acc1[8];
  }
  if (alu90) {
    data0_27300[alu5] = acc1[10];
  }
  if (alu90) {
    data0_27300[alu6] = acc1[12];
  }
  if (alu90) {
    data0_27300[alu7] = acc1[1];
  }
  if (alu90) {
    data0_27300[alu8] = acc1[3];
  }
  if (alu90) {
    data0_27300[alu9] = acc1[5];
  }
  if (alu90) {
    data0_27300[alu10] = acc1[7];
  }
  if (alu90) {
    data0_27300[alu11] = acc1[9];
  }
  if (alu90) {
    data0_27300[alu12] = acc1[11];
  }
  if (alu90) {
    data0_27300[alu13] = acc1[13];
  }
  if (alu90) {
    data0_27300[alu0] = acc1[0];
  }
}`;

const r_100_28_3_975 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<i32,84>;
@group(0) @binding(1)var<storage,read_write>data0_300:array<i32>;
@group(0) @binding(2)var<storage,read_write>data1_27300:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_32768:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_27300:array<i32>;
@group(0) @binding(5)var<storage,read_write>data4_27300:array<i32>;
@compute @workgroup_size(28) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<i32,3>;
  var acc1: array<i32,3>;
  var gidx0 = i32(gindex.x); /* 100 */
  var alu0 = (gidx0*3);
  var alu1 = (alu0+1);
  var val0 = data4_27300[alu1];
  var alu2 = (alu0+2);
  var val1 = data4_27300[alu2];
  var val2 = data4_27300[alu0];
  var val3 = data2_32768[alu1];
  var val4 = data2_32768[alu2];
  var val5 = data2_32768[alu0];
  var lidx0 = i32(lindex.x); /* 28 */
  acc0[0] = 0;
  acc0[1] = 0;
  acc0[2] = 0;
  for (var Ridx0 = 0; Ridx0 < 975; Ridx0++) {
    var alu6 = ((lidx0*975)+Ridx0);
    var val6 = data3_27300[alu6];
    var val7 = data1_27300[alu6];
    acc0[0] = (acc0[0]+((i32(((val7==val5)&(val6==val2))))*alu6));
    acc0[1] = (acc0[1]+((i32(((val7==val3)&(val6==val0))))*alu6));
    acc0[2] = (acc0[2]+((i32(((val7==val4)&(val6==val1))))*alu6));
  }
  var alu11 = (lidx0*3);
  temp0[(alu11+1)] = acc0[1];
  temp0[(alu11+2)] = acc0[2];
  temp0[alu11] = acc0[0];
  workgroupBarrier();
  acc1[0] = 0;
  acc1[1] = 0;
  acc1[2] = 0;
  for (var Ridx102 = 0; Ridx102 < 28; Ridx102++) {
    var alu19 = (Ridx102*3);
    var val8 = temp0[alu19];
    var val9 = temp0[(alu19+1)];
    var val10 = temp0[(alu19+2)];
    acc1[0] = (acc1[0]+val8);
    acc1[1] = (acc1[1]+val9);
    acc1[2] = (acc1[2]+val10);
  }
  var alu24 = (lidx0==0);
  if (alu24) {
    data0_300[alu1] = acc1[1];
  }
  if (alu24) {
    data0_300[alu2] = acc1[2];
  }
  if (alu24) {
    data0_300[alu0] = acc1[0];
  }
}`;

const E_300_6 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_1800:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_300:array<i32>;
@group(0) @binding(3)var<storage,read_write>data2_2400:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_15600:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_1200:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_32768:array<f32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 300 */
  var val0 = data1_300[gidx0];
  var alu0 = (val0/91);
  var alu1 = select(alu0,(alu0+-1),((val0<0)&((val0-(91*alu0))!=0)));
  var cast0 = bitcast<i32>((bitcast<u32>(alu1)<<2u));
  var alu2 = ((-1<alu1)&(alu1<300));
  var val1 = select(0.0f, data2_2400[(cast0+1200)], alu2);
  var val2 = select(0.0f, data3_15600[cast0], alu2);
  var alu3 = (cast0+2);
  var val3 = select(0.0f, data3_15600[alu3], alu2);
  var val4 = select(0.0f, data4_1200[cast0], alu2);
  var val5 = select(0.0f, data4_1200[alu3], alu2);
  var val6 = select(0.0f, data2_2400[(cast0+1201)], alu2);
  var val7 = select(0.0f, data2_2400[(cast0+1202)], alu2);
  var alu4 = (cast0+3);
  var val8 = select(0.0f, data3_15600[alu4], alu2);
  var alu5 = (cast0+1);
  var val9 = select(0.0f, data4_1200[alu5], alu2);
  var val10 = select(0.0f, data4_1200[alu4], alu2);
  var val11 = select(0.0f, data3_15600[alu5], alu2);
  var val12 = select(0.0f, data2_2400[(cast0+1203)], alu2);
  var val13 = data5_32768[gidx0];
  var alu6 = (gidx0*6);
  var alu7 = (exp2((val3*1.4426950408889634f))*val5);
  var alu8 = ((val1*alu7)+(val2*val5)+val4);
  var alu9 = (exp2((val7*1.4426950408889634f))*alu7);
  var alu10 = select(alu9,0.0f,(alu9<0.0f));
  var alu11 = (exp2((val8*1.4426950408889634f))*val10);
  var alu12 = ((val6*alu11)+(val11*val10)+val9);
  var alu13 = (exp2((val12*1.4426950408889634f))*alu11);
  var alu14 = select(alu13,0.0f,(alu13<0.0f));
  var alu15 = select(0.0f,(alu8+(alu10*-0.5f)),alu2);
  var alu16 = select(0.0f,(alu8+(0.5f*alu10)),alu2);
  var alu17 = select(0.0f,(alu12+(alu14*-0.5f)),alu2);
  var alu18 = select(0.0f,(alu12+(0.5f*alu14)),alu2);
  data0_1800[(alu6+1)] = alu17;
  data0_1800[(alu6+2)] = alu16;
  data0_1800[(alu6+3)] = alu18;
  data0_1800[(alu6+4)] = val13;
  data0_1800[(alu6+5)] = (f32((val0+(alu1*-91))));
  data0_1800[alu6] = alu15;
}`;

const E_300_6n1 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_1800:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_1800:array<f32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 300 */
  var alu0 = (gidx0*6);
  var alu1 = (alu0+1);
  var val0 = data1_1800[alu1];
  var alu2 = (alu0+2);
  var val1 = data1_1800[alu2];
  var alu3 = (alu0+3);
  var val2 = data1_1800[alu3];
  var alu4 = (alu0+4);
  var val3 = data1_1800[alu4];
  var alu5 = (alu0+5);
  var val4 = data1_1800[alu5];
  var val5 = data1_1800[alu0];
  data0_1800[alu1] = (val0*384.0f);
  data0_1800[alu2] = (val1*384.0f);
  data0_1800[alu3] = (val2*384.0f);
  data0_1800[alu4] = val3;
  data0_1800[alu5] = val4;
  data0_1800[alu0] = (val5*384.0f);
}`;

const setupNet = async (device, safetensor) => {
    const metadata = getTensorMetadata(safetensor);
    const infinityBuf = createInfinityUniformBuf(device);

    const layouts=[device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]})]

    const buf_0 = createEmptyBuf(device, 1769472);;
    const input0 = createEmptyBuf(device, 1769472);;
    const buf_1 = createEmptyBuf(device, 307200);;
    const buf_2 = createWeightBuf(device, 3993600, getTensorBuffer(safetensor, metadata['query_feat']));
    const buf_3 = createWeightBuf(device, 786432, getTensorBuffer(safetensor, metadata['transformer.decoder.layers.0.in_proj_weight']));
    const buf_4 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['transformer.decoder.layers.0.in_proj_bias']));
    const buf_5 = createEmptyBuf(device, 1769472);;
    const buf_6 = createWeightBuf(device, 12, getTensorBuffer(safetensor, metadata['means']));
    const buf_7 = createWeightBuf(device, 12, getTensorBuffer(safetensor, metadata['stds']));
    const buf_8 = createEmptyBuf(device, 886272);;
    const buf_9 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.embeddings.cls_token']));
    const buf_10 = createWeightBuf(device, 1179648, getTensorBuffer(safetensor, metadata['backbone.encoder.embeddings.projection.weight']));
    const buf_11 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.embeddings.projection.bias']));
    const buf_12 = createWeightBuf(device, 886272, getTensorBuffer(safetensor, metadata['backbone.encoder.embeddings.position_embeddings']));
    const buf_13 = createEmptyBuf(device, 2320);;
    const buf_14 = createEmptyBuf(device, 2320);;
    const buf_15 = createEmptyBuf(device, 890880);;
    const buf_16 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.0.norm1.weight']));
    const buf_17 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.0.norm1.bias']));
    const buf_18 = createEmptyBuf(device, 890880);;
    const buf_19 = createWeightBuf(device, 589824, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.0.attention.attention.query.weight']));
    const buf_20 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.0.attention.attention.query.bias']));
    const buf_21 = createEmptyBuf(device, 890880);;
    const buf_22 = createWeightBuf(device, 589824, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.0.attention.attention.key.weight']));
    const buf_23 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.0.attention.attention.key.bias']));
    const buf_24 = createEmptyBuf(device, 890880);;
    const buf_25 = createWeightBuf(device, 589824, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.0.attention.attention.value.weight']));
    const buf_26 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.0.attention.attention.value.bias']));
    const buf_27 = createEmptyBuf(device, 2018400);;
    const buf_28 = createEmptyBuf(device, 13920);;
    const buf_29 = createEmptyBuf(device, 13920);;
    const buf_30 = createWeightBuf(device, 589824, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.0.attention.dense.weight']));
    const buf_31 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.0.attention.dense.bias']));
    const buf_32 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.0.lambda1']));
    const buf_33 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.0.norm2.weight']));
    const buf_34 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.0.norm2.bias']));
    const buf_35 = createEmptyBuf(device, 3563520);;
    const buf_36 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.0.mlp.fc1.weight']));
    const buf_37 = createWeightBuf(device, 6144, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.0.mlp.fc1.bias']));
    const buf_38 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.0.mlp.fc2.weight']));
    const buf_39 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.0.mlp.fc2.bias']));
    const buf_40 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.0.lambda2']));
    const buf_41 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.1.norm1.weight']));
    const buf_42 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.1.norm1.bias']));
    const buf_43 = createWeightBuf(device, 589824, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.1.attention.attention.query.weight']));
    const buf_44 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.1.attention.attention.query.bias']));
    const buf_45 = createWeightBuf(device, 589824, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.1.attention.attention.key.weight']));
    const buf_46 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.1.attention.attention.key.bias']));
    const buf_47 = createEmptyBuf(device, 890880);;
    const buf_48 = createWeightBuf(device, 589824, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.1.attention.attention.value.weight']));
    const buf_49 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.1.attention.attention.value.bias']));
    const buf_50 = createWeightBuf(device, 589824, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.1.attention.dense.weight']));
    const buf_51 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.1.attention.dense.bias']));
    const buf_52 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.1.lambda1']));
    const buf_53 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.1.norm2.weight']));
    const buf_54 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.1.norm2.bias']));
    const buf_55 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.1.mlp.fc1.weight']));
    const buf_56 = createWeightBuf(device, 6144, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.1.mlp.fc1.bias']));
    const buf_57 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.1.mlp.fc2.weight']));
    const buf_58 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.1.mlp.fc2.bias']));
    const buf_59 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.1.lambda2']));
    const buf_60 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.2.norm1.weight']));
    const buf_61 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.2.norm1.bias']));
    const buf_62 = createWeightBuf(device, 589824, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.2.attention.attention.query.weight']));
    const buf_63 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.2.attention.attention.query.bias']));
    const buf_64 = createWeightBuf(device, 589824, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.2.attention.attention.key.weight']));
    const buf_65 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.2.attention.attention.key.bias']));
    const buf_66 = createWeightBuf(device, 589824, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.2.attention.attention.value.weight']));
    const buf_67 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.2.attention.attention.value.bias']));
    const buf_68 = createWeightBuf(device, 589824, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.2.attention.dense.weight']));
    const buf_69 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.2.attention.dense.bias']));
    const buf_70 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.2.lambda1']));
    const buf_71 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.2.norm2.weight']));
    const buf_72 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.2.norm2.bias']));
    const buf_73 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.2.mlp.fc1.weight']));
    const buf_74 = createWeightBuf(device, 6144, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.2.mlp.fc1.bias']));
    const buf_75 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.2.mlp.fc2.weight']));
    const buf_76 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.2.mlp.fc2.bias']));
    const buf_77 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.2.lambda2']));
    const buf_78 = createEmptyBuf(device, 2320);;
    const buf_79 = createEmptyBuf(device, 2320);;
    const buf_80 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.3.norm1.weight']));
    const buf_81 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.3.norm1.bias']));
    const buf_82 = createEmptyBuf(device, 884736);;
    const buf_83 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.layernorm.weight']));
    const buf_84 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.layernorm.bias']));
    const buf_85 = createWeightBuf(device, 589824, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.3.attention.attention.query.weight']));
    const buf_86 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.3.attention.attention.query.bias']));
    const buf_87 = createWeightBuf(device, 589824, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.3.attention.attention.key.weight']));
    const buf_88 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.3.attention.attention.key.bias']));
    const buf_89 = createWeightBuf(device, 589824, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.3.attention.attention.value.weight']));
    const buf_90 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.3.attention.attention.value.bias']));
    const buf_91 = createEmptyBuf(device, 8073600);;
    const buf_92 = createWeightBuf(device, 589824, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.3.attention.dense.weight']));
    const buf_93 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.3.attention.dense.bias']));
    const buf_94 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.3.lambda1']));
    const buf_95 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.3.norm2.weight']));
    const buf_96 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.3.norm2.bias']));
    const buf_97 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.3.mlp.fc1.weight']));
    const buf_98 = createWeightBuf(device, 6144, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.3.mlp.fc1.bias']));
    const buf_99 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.3.mlp.fc2.weight']));
    const buf_100 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.3.mlp.fc2.bias']));
    const buf_101 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.3.lambda2']));
    const buf_102 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.4.norm1.weight']));
    const buf_103 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.4.norm1.bias']));
    const buf_104 = createWeightBuf(device, 589824, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.4.attention.attention.query.weight']));
    const buf_105 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.4.attention.attention.query.bias']));
    const buf_106 = createWeightBuf(device, 589824, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.4.attention.attention.key.weight']));
    const buf_107 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.4.attention.attention.key.bias']));
    const buf_108 = createWeightBuf(device, 589824, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.4.attention.attention.value.weight']));
    const buf_109 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.4.attention.attention.value.bias']));
    const buf_110 = createWeightBuf(device, 589824, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.4.attention.dense.weight']));
    const buf_111 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.4.attention.dense.bias']));
    const buf_112 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.4.lambda1']));
    const buf_113 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.4.norm2.weight']));
    const buf_114 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.4.norm2.bias']));
    const buf_115 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.4.mlp.fc1.weight']));
    const buf_116 = createWeightBuf(device, 6144, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.4.mlp.fc1.bias']));
    const buf_117 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.4.mlp.fc2.weight']));
    const buf_118 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.4.mlp.fc2.bias']));
    const buf_119 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.4.lambda2']));
    const buf_120 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.5.norm1.weight']));
    const buf_121 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.5.norm1.bias']));
    const buf_122 = createWeightBuf(device, 589824, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.5.attention.attention.query.weight']));
    const buf_123 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.5.attention.attention.query.bias']));
    const buf_124 = createWeightBuf(device, 589824, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.5.attention.attention.key.weight']));
    const buf_125 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.5.attention.attention.key.bias']));
    const buf_126 = createWeightBuf(device, 589824, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.5.attention.attention.value.weight']));
    const buf_127 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.5.attention.attention.value.bias']));
    const buf_128 = createWeightBuf(device, 589824, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.5.attention.dense.weight']));
    const buf_129 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.5.attention.dense.bias']));
    const buf_130 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.5.lambda1']));
    const buf_131 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.5.norm2.weight']));
    const buf_132 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.5.norm2.bias']));
    const buf_133 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.5.mlp.fc1.weight']));
    const buf_134 = createWeightBuf(device, 6144, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.5.mlp.fc1.bias']));
    const buf_135 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.5.mlp.fc2.weight']));
    const buf_136 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.5.mlp.fc2.bias']));
    const buf_137 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.5.lambda2']));
    const buf_138 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.6.norm1.weight']));
    const buf_139 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.6.norm1.bias']));
    const buf_140 = createEmptyBuf(device, 884736);;
    const buf_141 = createWeightBuf(device, 589824, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.6.attention.attention.query.weight']));
    const buf_142 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.6.attention.attention.query.bias']));
    const buf_143 = createWeightBuf(device, 589824, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.6.attention.attention.key.weight']));
    const buf_144 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.6.attention.attention.key.bias']));
    const buf_145 = createWeightBuf(device, 589824, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.6.attention.attention.value.weight']));
    const buf_146 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.6.attention.attention.value.bias']));
    const buf_147 = createWeightBuf(device, 589824, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.6.attention.dense.weight']));
    const buf_148 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.6.attention.dense.bias']));
    const buf_149 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.6.lambda1']));
    const buf_150 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.6.norm2.weight']));
    const buf_151 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.6.norm2.bias']));
    const buf_152 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.6.mlp.fc1.weight']));
    const buf_153 = createWeightBuf(device, 6144, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.6.mlp.fc1.bias']));
    const buf_154 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.6.mlp.fc2.weight']));
    const buf_155 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.6.mlp.fc2.bias']));
    const buf_156 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.6.lambda2']));
    const buf_157 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.7.norm1.weight']));
    const buf_158 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.7.norm1.bias']));
    const buf_159 = createWeightBuf(device, 589824, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.7.attention.attention.query.weight']));
    const buf_160 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.7.attention.attention.query.bias']));
    const buf_161 = createWeightBuf(device, 589824, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.7.attention.attention.key.weight']));
    const buf_162 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.7.attention.attention.key.bias']));
    const buf_163 = createWeightBuf(device, 589824, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.7.attention.attention.value.weight']));
    const buf_164 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.7.attention.attention.value.bias']));
    const buf_165 = createWeightBuf(device, 589824, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.7.attention.dense.weight']));
    const buf_166 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.7.attention.dense.bias']));
    const buf_167 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.7.lambda1']));
    const buf_168 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.7.norm2.weight']));
    const buf_169 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.7.norm2.bias']));
    const buf_170 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.7.mlp.fc1.weight']));
    const buf_171 = createWeightBuf(device, 6144, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.7.mlp.fc1.bias']));
    const buf_172 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.7.mlp.fc2.weight']));
    const buf_173 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.7.mlp.fc2.bias']));
    const buf_174 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.7.lambda2']));
    const buf_175 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.8.norm1.weight']));
    const buf_176 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.8.norm1.bias']));
    const buf_177 = createWeightBuf(device, 589824, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.8.attention.attention.query.weight']));
    const buf_178 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.8.attention.attention.query.bias']));
    const buf_179 = createWeightBuf(device, 589824, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.8.attention.attention.key.weight']));
    const buf_180 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.8.attention.attention.key.bias']));
    const buf_181 = createWeightBuf(device, 589824, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.8.attention.attention.value.weight']));
    const buf_182 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.8.attention.attention.value.bias']));
    const buf_183 = createWeightBuf(device, 589824, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.8.attention.dense.weight']));
    const buf_184 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.8.attention.dense.bias']));
    const buf_185 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.8.lambda1']));
    const buf_186 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.8.norm2.weight']));
    const buf_187 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.8.norm2.bias']));
    const buf_188 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.8.mlp.fc1.weight']));
    const buf_189 = createWeightBuf(device, 6144, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.8.mlp.fc1.bias']));
    const buf_190 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.8.mlp.fc2.weight']));
    const buf_191 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.8.mlp.fc2.bias']));
    const buf_192 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.8.lambda2']));
    const buf_193 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.9.norm1.weight']));
    const buf_194 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.9.norm1.bias']));
    const buf_195 = createEmptyBuf(device, 884736);;
    const buf_196 = createWeightBuf(device, 589824, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.9.attention.attention.query.weight']));
    const buf_197 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.9.attention.attention.query.bias']));
    const buf_198 = createWeightBuf(device, 589824, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.9.attention.attention.key.weight']));
    const buf_199 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.9.attention.attention.key.bias']));
    const buf_200 = createWeightBuf(device, 589824, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.9.attention.attention.value.weight']));
    const buf_201 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.9.attention.attention.value.bias']));
    const buf_202 = createWeightBuf(device, 589824, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.9.attention.dense.weight']));
    const buf_203 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.9.attention.dense.bias']));
    const buf_204 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.9.lambda1']));
    const buf_205 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.9.norm2.weight']));
    const buf_206 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.9.norm2.bias']));
    const buf_207 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.9.mlp.fc1.weight']));
    const buf_208 = createWeightBuf(device, 6144, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.9.mlp.fc1.bias']));
    const buf_209 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.9.mlp.fc2.weight']));
    const buf_210 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.9.mlp.fc2.bias']));
    const buf_211 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.9.lambda2']));
    const buf_212 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.10.norm1.weight']));
    const buf_213 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.10.norm1.bias']));
    const buf_214 = createWeightBuf(device, 589824, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.10.attention.attention.query.weight']));
    const buf_215 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.10.attention.attention.query.bias']));
    const buf_216 = createWeightBuf(device, 589824, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.10.attention.attention.key.weight']));
    const buf_217 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.10.attention.attention.key.bias']));
    const buf_218 = createWeightBuf(device, 589824, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.10.attention.attention.value.weight']));
    const buf_219 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.10.attention.attention.value.bias']));
    const buf_220 = createWeightBuf(device, 589824, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.10.attention.dense.weight']));
    const buf_221 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.10.attention.dense.bias']));
    const buf_222 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.10.lambda1']));
    const buf_223 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.10.norm2.weight']));
    const buf_224 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.10.norm2.bias']));
    const buf_225 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.10.mlp.fc1.weight']));
    const buf_226 = createWeightBuf(device, 6144, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.10.mlp.fc1.bias']));
    const buf_227 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.10.mlp.fc2.weight']));
    const buf_228 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.10.mlp.fc2.bias']));
    const buf_229 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.10.lambda2']));
    const buf_230 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.11.norm1.weight']));
    const buf_231 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.11.norm1.bias']));
    const buf_232 = createWeightBuf(device, 589824, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.11.attention.attention.query.weight']));
    const buf_233 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.11.attention.attention.query.bias']));
    const buf_234 = createWeightBuf(device, 589824, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.11.attention.attention.key.weight']));
    const buf_235 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.11.attention.attention.key.bias']));
    const buf_236 = createWeightBuf(device, 589824, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.11.attention.attention.value.weight']));
    const buf_237 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.11.attention.attention.value.bias']));
    const buf_238 = createWeightBuf(device, 589824, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.11.attention.dense.weight']));
    const buf_239 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.11.attention.dense.bias']));
    const buf_240 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.11.lambda1']));
    const buf_241 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.11.norm2.weight']));
    const buf_242 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.11.norm2.bias']));
    const buf_243 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.11.mlp.fc1.weight']));
    const buf_244 = createWeightBuf(device, 6144, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.11.mlp.fc1.bias']));
    const buf_245 = createWeightBuf(device, 2359296, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.11.mlp.fc2.weight']));
    const buf_246 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.11.mlp.fc2.bias']));
    const buf_247 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['backbone.encoder.encoder.layer.11.lambda2']));
    const buf_248 = createEmptyBuf(device, 884736);;
    const buf_249 = createEmptyBuf(device, 3538944);;
    const buf_250 = createEmptyBuf(device, 589824);;
    const buf_251 = createWeightBuf(device, 1572864, getTensorBuffer(safetensor, metadata['backbone.projector.stages.0.0.cv1.conv.weight']));
    const buf_252 = createEmptyBuf(device, 2304);;
    const buf_253 = createEmptyBuf(device, 589824);;
    const buf_254 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['backbone.projector.stages.0.0.cv1.bn.weight']));
    const buf_255 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['backbone.projector.stages.0.0.cv1.bn.bias']));
    const buf_256 = createEmptyBuf(device, 294912);;
    const buf_257 = createWeightBuf(device, 589824, getTensorBuffer(safetensor, metadata['backbone.projector.stages.0.0.m.0.cv1.conv.weight']));
    const buf_258 = createEmptyBuf(device, 294912);;
    const buf_259 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['backbone.projector.stages.0.0.m.0.cv1.bn.weight']));
    const buf_260 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['backbone.projector.stages.0.0.m.0.cv1.bn.bias']));
    const buf_261 = createWeightBuf(device, 589824, getTensorBuffer(safetensor, metadata['backbone.projector.stages.0.0.m.0.cv2.conv.weight']));
    const buf_262 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['backbone.projector.stages.0.0.m.0.cv2.bn.weight']));
    const buf_263 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['backbone.projector.stages.0.0.m.0.cv2.bn.bias']));
    const buf_264 = createWeightBuf(device, 589824, getTensorBuffer(safetensor, metadata['backbone.projector.stages.0.0.m.1.cv1.conv.weight']));
    const buf_265 = createEmptyBuf(device, 294912);;
    const buf_266 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['backbone.projector.stages.0.0.m.1.cv1.bn.weight']));
    const buf_267 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['backbone.projector.stages.0.0.m.1.cv1.bn.bias']));
    const buf_268 = createWeightBuf(device, 589824, getTensorBuffer(safetensor, metadata['backbone.projector.stages.0.0.m.1.cv2.conv.weight']));
    const buf_269 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['backbone.projector.stages.0.0.m.1.cv2.bn.weight']));
    const buf_270 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['backbone.projector.stages.0.0.m.1.cv2.bn.bias']));
    const buf_271 = createWeightBuf(device, 589824, getTensorBuffer(safetensor, metadata['backbone.projector.stages.0.0.m.2.cv1.conv.weight']));
    const buf_272 = createEmptyBuf(device, 294912);;
    const buf_273 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['backbone.projector.stages.0.0.m.2.cv1.bn.weight']));
    const buf_274 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['backbone.projector.stages.0.0.m.2.cv1.bn.bias']));
    const buf_275 = createWeightBuf(device, 589824, getTensorBuffer(safetensor, metadata['backbone.projector.stages.0.0.m.2.cv2.conv.weight']));
    const buf_276 = createEmptyBuf(device, 1474560);;
    const buf_277 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['backbone.projector.stages.0.0.m.2.cv2.bn.weight']));
    const buf_278 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['backbone.projector.stages.0.0.m.2.cv2.bn.bias']));
    const buf_279 = createWeightBuf(device, 655360, getTensorBuffer(safetensor, metadata['backbone.projector.stages.0.0.cv2.conv.weight']));
    const buf_280 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['backbone.projector.stages.0.0.cv2.bn.weight']));
    const buf_281 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['backbone.projector.stages.0.0.cv2.bn.bias']));
    const buf_282 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['backbone.projector.stages.0.1.weight']));
    const buf_283 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['backbone.projector.stages.0.1.bias']));
    const buf_284 = createWeightBuf(device, 262144, getTensorBuffer(safetensor, metadata['transformer.decoder.layers.0.cross_attn.value_proj.weight']));
    const buf_285 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['transformer.decoder.layers.0.cross_attn.value_proj.bias']));
    const buf_286 = createEmptyBuf(device, 589824);;
    const buf_287 = createWeightBuf(device, 262144, getTensorBuffer(safetensor, metadata['transformer.enc_output.weight']));
    const buf_288 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['transformer.enc_output.bias']));
    const buf_289 = createEmptyBuf(device, 589824);;
    const buf_290 = createWeightBuf(device, 262144, getTensorBuffer(safetensor, metadata['transformer.decoder.layers.1.cross_attn.value_proj.weight']));
    const buf_291 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['transformer.decoder.layers.1.cross_attn.value_proj.bias']));
    const buf_292 = createEmptyBuf(device, 2304);;
    const buf_293 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['transformer.enc_output_norm.weight']));
    const buf_294 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['transformer.enc_output_norm.bias']));
    const buf_295 = createWeightBuf(device, 93184, getTensorBuffer(safetensor, metadata['transformer.enc_out_class_embed_w']));
    const buf_296 = createWeightBuf(device, 364, getTensorBuffer(safetensor, metadata['transformer.enc_out_class_embed_b']));
    const buf_297 = createWeightBuf(device, 262144, getTensorBuffer(safetensor, metadata['transformer.enc_out_bbox_embed.0.layers.0.weight']));
    const buf_298 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['transformer.enc_out_bbox_embed.0.layers.0.bias']));
    const buf_299 = createEmptyBuf(device, 4096);;
    const buf_300 = createEmptyBuf(device, 2304);;
    const buf_301 = createWeightBuf(device, 262144, getTensorBuffer(safetensor, metadata['transformer.enc_out_bbox_embed.0.layers.1.weight']));
    const buf_302 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['transformer.enc_out_bbox_embed.0.layers.1.bias']));
    const buf_303 = createEmptyBuf(device, 4096);;
    const buf_304 = createEmptyBuf(device, 9216);;
    const buf_305 = createWeightBuf(device, 4096, getTensorBuffer(safetensor, metadata['transformer.enc_out_bbox_embed.0.layers.2.weight']));
    const buf_306 = createWeightBuf(device, 16, getTensorBuffer(safetensor, metadata['transformer.enc_out_bbox_embed.0.layers.2.bias']));
    const buf_307 = createEmptyBuf(device, 2304);;
    const buf_308 = createEmptyBuf(device, 2304);;
    const buf_309 = createEmptyBuf(device, 4800);;
    const buf_310 = createEmptyBuf(device, 307200);;
    const buf_311 = createWeightBuf(device, 62400, getTensorBuffer(safetensor, metadata['refpoint_embed']));
    const buf_312 = createWeightBuf(device, 524288, getTensorBuffer(safetensor, metadata['transformer.decoder.ref_point_head.layers.0.weight']));
    const buf_313 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['transformer.decoder.ref_point_head.layers.0.bias']));
    const buf_314 = createEmptyBuf(device, 307200);;
    const buf_315 = createWeightBuf(device, 262144, getTensorBuffer(safetensor, metadata['transformer.decoder.ref_point_head.layers.1.weight']));
    const buf_316 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['transformer.decoder.ref_point_head.layers.1.bias']));
    const buf_317 = createEmptyBuf(device, 307200);;
    const buf_318 = createEmptyBuf(device, 2880000);;
    const buf_319 = createEmptyBuf(device, 9600);;
    const buf_320 = createEmptyBuf(device, 9600);;
    const buf_321 = createWeightBuf(device, 262144, getTensorBuffer(safetensor, metadata['transformer.decoder.layers.0.out_proj_weight']));
    const buf_322 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['transformer.decoder.layers.0.out_proj_bias']));
    const buf_323 = createEmptyBuf(device, 1200);;
    const buf_324 = createEmptyBuf(device, 1200);;
    const buf_325 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['transformer.decoder.layers.0.norm1.weight']));
    const buf_326 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['transformer.decoder.layers.0.norm1.bias']));
    const buf_327 = createEmptyBuf(device, 76800);;
    const buf_328 = createWeightBuf(device, 65536, getTensorBuffer(safetensor, metadata['transformer.decoder.layers.0.cross_attn.sampling_offsets.weight']));
    const buf_329 = createWeightBuf(device, 256, getTensorBuffer(safetensor, metadata['transformer.decoder.layers.0.cross_attn.sampling_offsets.bias']));
    const buf_330 = createEmptyBuf(device, 38400);;
    const buf_331 = createWeightBuf(device, 32768, getTensorBuffer(safetensor, metadata['transformer.decoder.layers.0.cross_attn.attention_weights.weight']));
    const buf_332 = createWeightBuf(device, 128, getTensorBuffer(safetensor, metadata['transformer.decoder.layers.0.cross_attn.attention_weights.bias']));
    const buf_333 = createEmptyBuf(device, 19200);;
    const buf_334 = createEmptyBuf(device, 19200);;
    const buf_335 = createWeightBuf(device, 262144, getTensorBuffer(safetensor, metadata['transformer.decoder.layers.0.cross_attn.output_proj.weight']));
    const buf_336 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['transformer.decoder.layers.0.cross_attn.output_proj.bias']));
    const buf_337 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['transformer.decoder.layers.0.norm2.weight']));
    const buf_338 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['transformer.decoder.layers.0.norm2.bias']));
    const buf_339 = createEmptyBuf(device, 2457600);;
    const buf_340 = createWeightBuf(device, 2097152, getTensorBuffer(safetensor, metadata['transformer.decoder.layers.0.linear1.weight']));
    const buf_341 = createWeightBuf(device, 8192, getTensorBuffer(safetensor, metadata['transformer.decoder.layers.0.linear1.bias']));
    const buf_342 = createWeightBuf(device, 2097152, getTensorBuffer(safetensor, metadata['transformer.decoder.layers.0.linear2.weight']));
    const buf_343 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['transformer.decoder.layers.0.linear2.bias']));
    const buf_344 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['transformer.decoder.layers.0.norm3.weight']));
    const buf_345 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['transformer.decoder.layers.0.norm3.bias']));
    const buf_346 = createWeightBuf(device, 786432, getTensorBuffer(safetensor, metadata['transformer.decoder.layers.1.in_proj_weight']));
    const buf_347 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['transformer.decoder.layers.1.in_proj_bias']));
    const buf_348 = createEmptyBuf(device, 307200);;
    const buf_349 = createEmptyBuf(device, 614400);;
    const buf_350 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['transformer.decoder.norm.weight']));
    const buf_351 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['transformer.decoder.norm.bias']));
    const buf_352 = createWeightBuf(device, 262144, getTensorBuffer(safetensor, metadata['transformer.decoder.layers.1.out_proj_weight']));
    const buf_353 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['transformer.decoder.layers.1.out_proj_bias']));
    const buf_354 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['transformer.decoder.layers.1.norm1.weight']));
    const buf_355 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['transformer.decoder.layers.1.norm1.bias']));
    const buf_356 = createWeightBuf(device, 65536, getTensorBuffer(safetensor, metadata['transformer.decoder.layers.1.cross_attn.sampling_offsets.weight']));
    const buf_357 = createWeightBuf(device, 256, getTensorBuffer(safetensor, metadata['transformer.decoder.layers.1.cross_attn.sampling_offsets.bias']));
    const buf_358 = createWeightBuf(device, 32768, getTensorBuffer(safetensor, metadata['transformer.decoder.layers.1.cross_attn.attention_weights.weight']));
    const buf_359 = createWeightBuf(device, 128, getTensorBuffer(safetensor, metadata['transformer.decoder.layers.1.cross_attn.attention_weights.bias']));
    const buf_360 = createWeightBuf(device, 262144, getTensorBuffer(safetensor, metadata['transformer.decoder.layers.1.cross_attn.output_proj.weight']));
    const buf_361 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['transformer.decoder.layers.1.cross_attn.output_proj.bias']));
    const buf_362 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['transformer.decoder.layers.1.norm2.weight']));
    const buf_363 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['transformer.decoder.layers.1.norm2.bias']));
    const buf_364 = createWeightBuf(device, 2097152, getTensorBuffer(safetensor, metadata['transformer.decoder.layers.1.linear1.weight']));
    const buf_365 = createWeightBuf(device, 8192, getTensorBuffer(safetensor, metadata['transformer.decoder.layers.1.linear1.bias']));
    const buf_366 = createWeightBuf(device, 2097152, getTensorBuffer(safetensor, metadata['transformer.decoder.layers.1.linear2.weight']));
    const buf_367 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['transformer.decoder.layers.1.linear2.bias']));
    const buf_368 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['transformer.decoder.layers.1.norm3.weight']));
    const buf_369 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['transformer.decoder.layers.1.norm3.bias']));
    const buf_370 = createEmptyBuf(device, 614400);;
    const buf_371 = createEmptyBuf(device, 614400);;
    const buf_372 = createEmptyBuf(device, 109200);;
    const buf_373 = createWeightBuf(device, 93184, getTensorBuffer(safetensor, metadata['class_embed.weight']));
    const buf_374 = createWeightBuf(device, 364, getTensorBuffer(safetensor, metadata['class_embed.bias']));
    const buf_375 = createWeightBuf(device, 262144, getTensorBuffer(safetensor, metadata['bbox_embed.layers.0.weight']));
    const buf_376 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['bbox_embed.layers.0.bias']));
    const buf_377 = createEmptyBuf(device, 131072);;
    const buf_378 = createEmptyBuf(device, 109200);;
    const buf_379 = createWeightBuf(device, 262144, getTensorBuffer(safetensor, metadata['bbox_embed.layers.1.weight']));
    const buf_380 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['bbox_embed.layers.1.bias']));
    const buf_381 = createEmptyBuf(device, 131072);;
    const buf_382 = createWeightBuf(device, 4096, getTensorBuffer(safetensor, metadata['bbox_embed.layers.2.weight']));
    const buf_383 = createWeightBuf(device, 16, getTensorBuffer(safetensor, metadata['bbox_embed.layers.2.bias']));
    const buf_384 = createEmptyBuf(device, 109200);;
    const buf_385 = createEmptyBuf(device, 1200);;
    const buf_386 = createEmptyBuf(device, 7200);;
    const output0 = createEmptyBuf(device, 7200);;

    const gpuWriteBuffer0 = device.createBuffer({size:input0.size, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.MAP_WRITE });

    const gpuReadBuffer0 = device.createBuffer({size:output0.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    const kernels = [E_4608_32_3, r_30_8_16_5_2_2_256, E_9216_3_16, E_192_24_16_3_2, E_384_12_16_3_2, r_577_48_16_2_4_16_3, r_2_2_145_32_12, r_2_2_145_32_12n1, E_2_145_24_16_2, r_20_48_29_4_2_384, r_20_48_29_4_2_384, r_20_48_29_4_2_384, r_4_6_5_29_29_5_64, r_120_29_145, r_145_8_3_145, r_4_145_16_4_6_145, r_2_2_5_96_29_4_384, r_580_16_24, r_580_16_24n1, E_290_24_16_2, r_20_128_29_4_3_384, r_5_32_29_4_3_4_1536, r_580_16_24, r_580_16_24n1, E_290_24_16_2, r_20_48_29_4_2_384, r_20_48_29_4_2_384, r_20_48_29_4_2_384, r_4_6_5_29_29_5_64, r_120_29_145, r_145_8_3_145, r_4_145_16_4_6_145, r_5_32_29_4_3_4_384, r_580_16_24, r_580_16_24n1, E_290_24_16_2, r_20_128_29_4_3_384, r_5_32_29_4_3_4_1536, r_580_16_24, r_580_16_24n1, E_290_24_16_2, r_20_48_29_4_2_384, r_20_48_29_4_2_384, r_20_48_29_4_2_384, r_4_6_5_29_29_5_64, r_120_29_145, r_145_8_3_145, r_4_145_16_4_6_145, r_5_32_29_4_3_4_384, r_580_16_24, r_580_16_24n1, E_290_24_16_2, r_20_128_29_4_3_384, r_5_32_29_4_3_4_1536, r_580_16_24n2, r_580_16_24, r_580_16_24n3, r_580_16_24n1, E_580_48_8, E_12_2_12_2_32_12, r_20_32_29_4_3_384, r_20_32_29_4_3_384, r_20_32_29_4_3_384, r_6_116_20_29_5_64, r_870_4_580, r_435_8_580, r_116_3_4_16_2_5_580, r_5_32_29_4_3_4_384n1, r_580_16_24, r_580_16_24n1, E_290_24_16_2, r_20_128_29_4_3_384, r_5_32_29_4_3_4_1536, r_580_16_24, r_580_16_24n1, E_290_24_16_2, r_20_48_29_4_2_384, r_20_48_29_4_2_384, r_20_48_29_4_2_384, r_4_6_5_29_29_5_64, r_120_29_145, r_145_8_3_145, r_4_145_16_4_6_145, r_5_32_29_4_3_4_384, r_580_16_24, r_580_16_24n1, E_290_24_16_2, r_20_128_29_4_3_384, r_5_32_29_4_3_4_1536, r_580_16_24, r_580_16_24n1, E_290_24_16_2, r_20_48_29_4_2_384, r_20_48_29_4_2_384, r_20_48_29_4_2_384, r_4_6_5_29_29_5_64, r_120_29_145, r_145_8_3_145, r_4_145_16_4_6_145, r_5_32_29_4_3_4_384, r_580_16_24, r_580_16_24n1, E_290_24_16_2, r_20_128_29_4_3_384, r_5_32_29_4_3_4_1536, r_580_16_24n2, r_580_16_24, r_580_16_24n3, r_580_16_24n1, E_580_48_8, E_12_2_12_2_32_12, r_20_32_29_4_3_384, r_20_32_29_4_3_384, r_20_32_29_4_3_384, r_6_116_20_29_5_64, r_870_4_580, r_435_8_580, r_116_3_4_16_2_5_580, r_5_32_29_4_3_4_384n1, r_580_16_24, r_580_16_24n1, E_290_24_16_2, r_20_128_29_4_3_384, r_5_32_29_4_3_4_1536, r_580_16_24, r_580_16_24n1, E_290_24_16_2, r_20_48_29_4_2_384, r_20_48_29_4_2_384, r_20_48_29_4_2_384, r_4_6_5_29_29_5_64, r_120_29_145, r_145_8_3_145, r_4_145_16_4_6_145, r_5_32_29_4_3_4_384, r_580_16_24, r_580_16_24n1, E_290_24_16_2, r_20_128_29_4_3_384, r_5_32_29_4_3_4_1536, r_580_16_24, r_580_16_24n1, E_290_24_16_2, r_20_48_29_4_2_384, r_20_48_29_4_2_384, r_20_48_29_4_2_384, r_4_6_5_29_29_5_64, r_120_29_145, r_145_8_3_145, r_4_145_16_4_6_145, r_5_32_29_4_3_4_384, r_580_16_24, r_580_16_24n1, E_290_24_16_2, r_20_128_29_4_3_384, r_5_32_29_4_3_4_1536, r_580_16_24n2, r_580_16_24, r_580_16_24n3, r_580_16_24n1, E_580_48_8, E_12_2_12_2_32_12, r_20_32_29_4_3_384, r_20_32_29_4_3_384, r_20_32_29_4_3_384, r_6_116_20_29_5_64, r_870_4_580, r_435_8_580, r_116_3_4_16_2_5_580, r_5_32_29_4_3_4_384n1, r_580_16_24, r_580_16_24n1, E_290_24_16_2, r_20_128_29_4_3_384, r_5_32_29_4_3_4_1536, r_580_16_24, r_580_16_24n1, E_290_24_16_2, r_20_48_29_4_2_384, r_20_48_29_4_2_384, r_20_48_29_4_2_384, r_4_6_5_29_29_5_64, r_120_29_145, r_145_8_3_145, r_4_145_16_4_6_145, r_5_32_29_4_3_4_384, r_580_16_24, r_580_16_24n1, E_290_24_16_2, r_20_128_29_4_3_384, r_5_32_29_4_3_4_1536, r_580_16_24, r_580_16_24n1, E_290_24_16_2, r_20_48_29_4_2_384, r_20_48_29_4_2_384, r_20_48_29_4_2_384, r_4_6_5_29_29_5_64, r_120_29_145, r_145_8_3_145, r_4_145_16_4_6_145, r_5_32_29_4_3_4_384, r_580_16_24, r_580_16_24n1, E_290_24_16_2, r_20_128_29_4_3_384, r_5_32_29_4_3_4_1536, r_580_16_24, r_580_16_24n1, E_12_2_12_2_32_12, E_9216_32_3, r_6_64_32_4_3_1536, r_576_16_16, E_576_32_8, r_576_16_16n1, E_32_192_8_3, r_6_24_8_16_4_128_3_3, r_144_4_128, E_72_128_8, r_576_32_4, E_8_576_16, r_24_24_32_16_4_8_3_3, r_144_4_128, E_72_128_8, r_576_32_4, E_8_576_16, r_24_24_32_16_4_8_3_3, r_144_4_128, E_72_128_8, r_576_32_4, E_8_576_16, r_24_24_32_16_4_8_3_3, r_144_4_128, E_72_128_8, r_576_32_4, E_8_576_16, r_24_24_32_16_4_8_3_3, r_144_4_128, E_72_128_8, r_576_32_4, E_8_576_16, r_24_24_32_16_4_8_3_3, r_144_4_128, E_72_128_8, r_576_32_4, E_160_36_16_2_2, r_6_64_32_3_4_160_4, r_576_16_16, E_576_32_8, r_576_16_16n1, E_576_16_16, r_576_16_16, E_576_32_8, r_576_16_16n1, E_576_16_16n1, r_8_8_24_32_3_256, r_24_2_16_2_8_12_24_24_4_64_4, r_8_8_24_32_3_256, r_576_16_16n2, r_576_16_16n3, E_72_64_8_4, r_48_91_3_4_256, r_144_16_16_4_256, E_256_2_2, r_576_32_18, r_144_16_16_4_256, E_512_2, r_576_4_16_16, E_128_2_2_2, E_256_2_2n1, E_512_2, E_64_2_4_2, E_128_4_2, E_256_2_2n1, E_512_2, E_32_2_8_2, E_64_8_2, E_128_4_2, E_256_2_2n1, E_512_2, E_2_2_16_16, E_32_16_2, E_64_8_2, E_128_4_2, E_256_2_2n1, E_512_2, E_2_2_32_8, E_2_32_16, E_32_16_2, E_64_8_2, E_128_4_2, E_256_2_2n1, E_512_2, E_2_2_64_4, E_2_64_8, E_2_32_16, E_32_16_2, E_64_8_2, E_128_4_2, E_256_2_2n1, E_512_2, E_2_2_128_2, E_2_128_4, E_2_64_8, E_2_32_16, E_32_16_2, E_64_8_2, E_128_4_2, E_256_2_2n1, E_512_2, E_2_256_2, E_2_256_2n1, E_2_128_4, E_2_64_8, E_2_32_16, E_32_16_2, E_64_8_2, E_128_4_2, E_256_2_2n1, E_512_2, E_512_2n1, E_2_256_2n1, E_2_128_4, E_2_64_8, E_2_32_16, E_32_16_2, E_64_8_2, E_128_4_2, E_256_2_2n1, E_512_2, r_72_8_576, r_144_16_4_36, r_150_24_4_2_24_24_24_4, r_300_32_32_4_2_2_4_2, r_15_4_8_4_8_5_256, r_75_2_16_8_4_256, r_75_32_8_4_256, r_2_300_75_4_4_32, r_600_4_75_4, r_150_16_75_4, r_60_8_16_5_2_300, r_300_16_16_256, r_150_2_256, r_300_16_16, E_300_16_16, r_150_2_2_16_2_16_16, r_75_32_4_8_32, r_2400_2_2, r_2400_2_2n1, r_150_16_16_2_2, r_75_16_4_16_256, r_150_2_256, r_300_16_16, E_300_16_16, r_15_64_16_2_5_2_2_256, r_60_8_16_5_2_2048, r_150_2_256, r_300_16_16, E_300_16_16, r_60_16_16_5_256, r_60_16_16_5_256n1, r_30_16_16_5_2_256, r_150_2_256, r_2_300_75_4_4_32, r_300_16_16, r_600_4_75_4, E_2_300_16_16, r_150_16_75_4, r_60_8_16_5_2_300, r_75_16_4_16_256, r_150_2_256, r_300_16_16, E_300_16_16, r_150_2_2_16_2_16_16, r_75_32_4_8_32, r_2400_2_2, r_2400_2_2n1, r_150_16_16_2_2, r_75_16_4_16_256, r_150_2_256, r_300_16_16, E_300_16_16, r_15_64_16_2_5_2_2_256, r_60_8_16_5_2_2048, r_150_2_256, r_300_16_16, E_300_16_16, r_150_2_256, r_300_16_16, E_2_300_16_16n1, E_4800_32, r_300_13_7_256, r_200_16_3_16_256, E_4_2_4_32_2_2_8, r_1950_28_7_2_975, r_200_16_3_16_256, E_1024_2_16, r_75_4_8_256, E_512_2_2_2_8, E_2048_2_2_4, E_1024_2_16, E_2048_2_2_4n1, E_1024_2_4_4, E_2048_2_2_4, E_1024_2_16, E_1024_2_2_2_4, E_2048_2_8, E_1024_2_4_4, E_2048_2_2_4, E_1024_2_16, E_512_2_16_2, E_1024_2_4_4n1, E_2048_2_8, E_1024_2_4_4, E_2048_2_2_4, E_1024_2_16, E_32_2_2_32_8, E_128_2_32_4, E_1024_2_4_4n1, E_2048_2_8, E_1024_2_4_4, E_2048_2_2_4, E_1024_2_16, E_16_2_2_64_8, E_256_2_16_4, E_128_2_32_4, E_1024_2_4_4n1, E_2048_2_8, E_1024_2_4_4, E_2048_2_2_4, E_1024_2_16, E_64_2_2_32_4, E_128_2_32_4n1, E_256_2_16_4, E_128_2_32_4, E_1024_2_4_4n1, E_2048_2_8, E_1024_2_4_4, E_2048_2_2_4, E_1024_2_16, E_8_2_2_256_4, E_64_2_64_4, E_128_2_32_4n1, E_256_2_16_4, E_128_2_32_4, E_1024_2_4_4n1, E_2048_2_8, E_1024_2_4_4, E_2048_2_2_4, E_1024_2_16, E_2_2_512_16, E_8_2_512_4, E_64_2_64_4, E_128_2_32_4n1, E_256_2_16_4, E_128_2_32_4, E_1024_2_4_4n1, E_2048_2_8, E_1024_2_4_4, E_2048_2_2_4, E_1024_2_16, E_2_2_2_1024_4, E_2_1024_16, E_8_2_512_4, E_64_2_64_4, E_128_2_32_4n1, E_256_2_16_4, E_128_2_32_4, E_1024_2_4_4n1, E_2048_2_8, E_1024_2_4_4, E_2048_2_2_4, E_1024_2_16, E_4_2_2048_2, E_2_2048_8, E_2_1024_16, E_8_2_512_4, E_64_2_64_4, E_128_2_32_4n1, E_256_2_16_4, E_128_2_32_4, E_1024_2_4_4n1, E_2048_2_8, E_1024_2_4_4, E_2048_2_2_4, E_1024_2_16, E_2_2_2_1024_4n1, E_2_4096_4, E_2_2048_8, E_2_1024_16, E_8_2_512_4, E_64_2_64_4, E_128_2_32_4n1, E_256_2_16_4, E_128_2_32_4, E_1024_2_4_4n1, E_2048_2_8, E_1024_2_4_4, E_2048_2_2_4, E_1024_2_16, E_2_2_2048_4, E_2_2_2048_4n1, E_2_4096_4, E_2_2048_8, E_2_1024_16, E_8_2_512_4, E_64_2_64_4, E_128_2_32_4n1, E_256_2_16_4, E_128_2_32_4, E_1024_2_4_4n1, E_2048_2_8, E_1024_2_4_4, E_2048_2_2_4, E_1024_2_16, E_2_4096_4n1, E_2_2_2048_4n1, E_2_4096_4, E_2_2048_8, E_2_1024_16, E_8_2_512_4, E_64_2_64_4, E_128_2_32_4n1, E_256_2_16_4, E_128_2_32_4, E_1024_2_4_4n1, E_2048_2_8, E_1024_2_4_4, E_2048_2_2_4, E_1024_2_16, r_1950_28_7_2_975n1, r_100_28_3_975, E_300_6, E_300_6n1];
    const pipelines = await Promise.all(kernels.map(async (name, i) => {
      return await device.createComputePipelineAsync({
          layout: device.createPipelineLayout({
              bindGroupLayouts: [layouts[i]],
          }),
          compute: {
              module: device.createShaderModule({
                  code: name,
              }),
              entryPoint: "main",
          },
      });
  }))

    return async (_input0) => {
        const commandEncoder = device.createCommandEncoder();
        await gpuWriteBuffer0.mapAsync(GPUMapMode.WRITE);
        new Float32Array(gpuWriteBuffer0.getMappedRange()).set(_input0);
        gpuWriteBuffer0.unmap();
        commandEncoder.copyBufferToBuffer(gpuWriteBuffer0, 0, input0, 0, gpuWriteBuffer0.size);
        addComputePass(device, commandEncoder, pipelines[0], layouts[0], infinityBuf, [buf_0, input0], [4608, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[1], layouts[1], infinityBuf, [buf_1, buf_2, buf_3, buf_4], [8, 30, 1]);
        addComputePass(device, commandEncoder, pipelines[2], layouts[2], infinityBuf, [input0, buf_0], [3, 9216, 1]);
        addComputePass(device, commandEncoder, pipelines[3], layouts[3], infinityBuf, [buf_0, input0], [24, 192, 1]);
        addComputePass(device, commandEncoder, pipelines[4], layouts[4], infinityBuf, [buf_5, buf_0, buf_6, buf_7], [12, 384, 1]);
        addComputePass(device, commandEncoder, pipelines[5], layouts[5], infinityBuf, [buf_8, buf_9, buf_5, buf_10, buf_11, buf_12], [48, 577, 1]);
        addComputePass(device, commandEncoder, pipelines[6], layouts[6], infinityBuf, [buf_13, buf_8], [145, 2, 2]);
        addComputePass(device, commandEncoder, pipelines[7], layouts[7], infinityBuf, [buf_14, buf_8, buf_13], [145, 2, 2]);
        addComputePass(device, commandEncoder, pipelines[8], layouts[8], infinityBuf, [buf_15, buf_8, buf_13, buf_14, buf_16, buf_17], [24, 145, 2]);
        addComputePass(device, commandEncoder, pipelines[9], layouts[9], infinityBuf, [buf_18, buf_15, buf_19, buf_20], [48, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[10], layouts[10], infinityBuf, [buf_21, buf_15, buf_22, buf_23], [48, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[11], layouts[11], infinityBuf, [buf_24, buf_15, buf_25, buf_26], [48, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[12], layouts[12], infinityBuf, [buf_27, buf_18, buf_21], [145, 6, 4]);
        addComputePass(device, commandEncoder, pipelines[13], layouts[13], infinityBuf, [buf_28, buf_27], [120, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[14], layouts[14], infinityBuf, [buf_29, buf_27, buf_28], [145, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[15], layouts[15], infinityBuf, [buf_21, buf_27, buf_28, buf_29, buf_24], [145, 4, 1]);
        addComputePass(device, commandEncoder, pipelines[16], layouts[16], infinityBuf, [buf_24, buf_21, buf_30, buf_31, buf_32, buf_8], [480, 2, 2]);
        addComputePass(device, commandEncoder, pipelines[17], layouts[17], infinityBuf, [buf_14, buf_24], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[18], layouts[18], infinityBuf, [buf_13, buf_24, buf_14], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[19], layouts[19], infinityBuf, [buf_21, buf_24, buf_14, buf_13, buf_33, buf_34], [24, 290, 1]);
        addComputePass(device, commandEncoder, pipelines[20], layouts[20], infinityBuf, [buf_35, buf_21, buf_36, buf_37], [128, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[21], layouts[21], infinityBuf, [buf_21, buf_35, buf_38, buf_39, buf_40, buf_24], [32, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[22], layouts[22], infinityBuf, [buf_13, buf_21], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[23], layouts[23], infinityBuf, [buf_14, buf_21, buf_13], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[24], layouts[24], infinityBuf, [buf_24, buf_21, buf_13, buf_14, buf_41, buf_42], [24, 290, 1]);
        addComputePass(device, commandEncoder, pipelines[25], layouts[25], infinityBuf, [buf_18, buf_24, buf_43, buf_44], [48, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[26], layouts[26], infinityBuf, [buf_15, buf_24, buf_45, buf_46], [48, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[27], layouts[27], infinityBuf, [buf_47, buf_24, buf_48, buf_49], [48, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[28], layouts[28], infinityBuf, [buf_27, buf_18, buf_15], [145, 6, 4]);
        addComputePass(device, commandEncoder, pipelines[29], layouts[29], infinityBuf, [buf_29, buf_27], [120, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[30], layouts[30], infinityBuf, [buf_28, buf_27, buf_29], [145, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[31], layouts[31], infinityBuf, [buf_15, buf_27, buf_29, buf_28, buf_47], [145, 4, 1]);
        addComputePass(device, commandEncoder, pipelines[32], layouts[32], infinityBuf, [buf_47, buf_15, buf_50, buf_51, buf_52, buf_21], [32, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[33], layouts[33], infinityBuf, [buf_14, buf_47], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[34], layouts[34], infinityBuf, [buf_13, buf_47, buf_14], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[35], layouts[35], infinityBuf, [buf_15, buf_47, buf_14, buf_13, buf_53, buf_54], [24, 290, 1]);
        addComputePass(device, commandEncoder, pipelines[36], layouts[36], infinityBuf, [buf_35, buf_15, buf_55, buf_56], [128, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[37], layouts[37], infinityBuf, [buf_15, buf_35, buf_57, buf_58, buf_59, buf_47], [32, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[38], layouts[38], infinityBuf, [buf_13, buf_15], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[39], layouts[39], infinityBuf, [buf_14, buf_15, buf_13], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[40], layouts[40], infinityBuf, [buf_47, buf_15, buf_13, buf_14, buf_60, buf_61], [24, 290, 1]);
        addComputePass(device, commandEncoder, pipelines[41], layouts[41], infinityBuf, [buf_21, buf_47, buf_62, buf_63], [48, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[42], layouts[42], infinityBuf, [buf_18, buf_47, buf_64, buf_65], [48, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[43], layouts[43], infinityBuf, [buf_24, buf_47, buf_66, buf_67], [48, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[44], layouts[44], infinityBuf, [buf_27, buf_21, buf_18], [145, 6, 4]);
        addComputePass(device, commandEncoder, pipelines[45], layouts[45], infinityBuf, [buf_28, buf_27], [120, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[46], layouts[46], infinityBuf, [buf_29, buf_27, buf_28], [145, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[47], layouts[47], infinityBuf, [buf_18, buf_27, buf_28, buf_29, buf_24], [145, 4, 1]);
        addComputePass(device, commandEncoder, pipelines[48], layouts[48], infinityBuf, [buf_24, buf_18, buf_68, buf_69, buf_70, buf_15], [32, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[49], layouts[49], infinityBuf, [buf_14, buf_24], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[50], layouts[50], infinityBuf, [buf_13, buf_24, buf_14], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[51], layouts[51], infinityBuf, [buf_18, buf_24, buf_14, buf_13, buf_71, buf_72], [24, 290, 1]);
        addComputePass(device, commandEncoder, pipelines[52], layouts[52], infinityBuf, [buf_35, buf_18, buf_73, buf_74], [128, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[53], layouts[53], infinityBuf, [buf_18, buf_35, buf_75, buf_76, buf_77, buf_24], [32, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[54], layouts[54], infinityBuf, [buf_13, buf_18], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[55], layouts[55], infinityBuf, [buf_14, buf_18], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[56], layouts[56], infinityBuf, [buf_78, buf_18, buf_13], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[57], layouts[57], infinityBuf, [buf_79, buf_18, buf_14], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[58], layouts[58], infinityBuf, [buf_24, buf_18, buf_13, buf_78, buf_80, buf_81], [48, 580, 1]);
        addComputePass(device, commandEncoder, pipelines[59], layouts[59], infinityBuf, [buf_82, buf_18, buf_14, buf_79, buf_83, buf_84], [24, 2, 12]);
        addComputePass(device, commandEncoder, pipelines[60], layouts[60], infinityBuf, [buf_15, buf_24, buf_85, buf_86], [32, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[61], layouts[61], infinityBuf, [buf_21, buf_24, buf_87, buf_88], [32, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[62], layouts[62], infinityBuf, [buf_47, buf_24, buf_89, buf_90], [32, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[63], layouts[63], infinityBuf, [buf_91, buf_15, buf_21], [20, 116, 6]);
        addComputePass(device, commandEncoder, pipelines[64], layouts[64], infinityBuf, [buf_29, buf_91], [870, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[65], layouts[65], infinityBuf, [buf_28, buf_91, buf_29], [435, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[66], layouts[66], infinityBuf, [buf_21, buf_91, buf_29, buf_28, buf_47], [4, 3, 116]);
        addComputePass(device, commandEncoder, pipelines[67], layouts[67], infinityBuf, [buf_47, buf_21, buf_92, buf_93, buf_94, buf_18], [32, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[68], layouts[68], infinityBuf, [buf_79, buf_47], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[69], layouts[69], infinityBuf, [buf_14, buf_47, buf_79], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[70], layouts[70], infinityBuf, [buf_21, buf_47, buf_79, buf_14, buf_95, buf_96], [24, 290, 1]);
        addComputePass(device, commandEncoder, pipelines[71], layouts[71], infinityBuf, [buf_35, buf_21, buf_97, buf_98], [128, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[72], layouts[72], infinityBuf, [buf_21, buf_35, buf_99, buf_100, buf_101, buf_47], [32, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[73], layouts[73], infinityBuf, [buf_14, buf_21], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[74], layouts[74], infinityBuf, [buf_79, buf_21, buf_14], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[75], layouts[75], infinityBuf, [buf_47, buf_21, buf_14, buf_79, buf_102, buf_103], [24, 290, 1]);
        addComputePass(device, commandEncoder, pipelines[76], layouts[76], infinityBuf, [buf_18, buf_47, buf_104, buf_105], [48, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[77], layouts[77], infinityBuf, [buf_15, buf_47, buf_106, buf_107], [48, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[78], layouts[78], infinityBuf, [buf_24, buf_47, buf_108, buf_109], [48, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[79], layouts[79], infinityBuf, [buf_27, buf_18, buf_15], [145, 6, 4]);
        addComputePass(device, commandEncoder, pipelines[80], layouts[80], infinityBuf, [buf_28, buf_27], [120, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[81], layouts[81], infinityBuf, [buf_29, buf_27, buf_28], [145, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[82], layouts[82], infinityBuf, [buf_15, buf_27, buf_28, buf_29, buf_24], [145, 4, 1]);
        addComputePass(device, commandEncoder, pipelines[83], layouts[83], infinityBuf, [buf_24, buf_15, buf_110, buf_111, buf_112, buf_21], [32, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[84], layouts[84], infinityBuf, [buf_79, buf_24], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[85], layouts[85], infinityBuf, [buf_14, buf_24, buf_79], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[86], layouts[86], infinityBuf, [buf_15, buf_24, buf_79, buf_14, buf_113, buf_114], [24, 290, 1]);
        addComputePass(device, commandEncoder, pipelines[87], layouts[87], infinityBuf, [buf_35, buf_15, buf_115, buf_116], [128, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[88], layouts[88], infinityBuf, [buf_15, buf_35, buf_117, buf_118, buf_119, buf_24], [32, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[89], layouts[89], infinityBuf, [buf_14, buf_15], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[90], layouts[90], infinityBuf, [buf_79, buf_15, buf_14], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[91], layouts[91], infinityBuf, [buf_24, buf_15, buf_14, buf_79, buf_120, buf_121], [24, 290, 1]);
        addComputePass(device, commandEncoder, pipelines[92], layouts[92], infinityBuf, [buf_21, buf_24, buf_122, buf_123], [48, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[93], layouts[93], infinityBuf, [buf_18, buf_24, buf_124, buf_125], [48, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[94], layouts[94], infinityBuf, [buf_47, buf_24, buf_126, buf_127], [48, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[95], layouts[95], infinityBuf, [buf_27, buf_21, buf_18], [145, 6, 4]);
        addComputePass(device, commandEncoder, pipelines[96], layouts[96], infinityBuf, [buf_29, buf_27], [120, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[97], layouts[97], infinityBuf, [buf_28, buf_27, buf_29], [145, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[98], layouts[98], infinityBuf, [buf_18, buf_27, buf_29, buf_28, buf_47], [145, 4, 1]);
        addComputePass(device, commandEncoder, pipelines[99], layouts[99], infinityBuf, [buf_47, buf_18, buf_128, buf_129, buf_130, buf_15], [32, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[100], layouts[100], infinityBuf, [buf_79, buf_47], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[101], layouts[101], infinityBuf, [buf_14, buf_47, buf_79], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[102], layouts[102], infinityBuf, [buf_18, buf_47, buf_79, buf_14, buf_131, buf_132], [24, 290, 1]);
        addComputePass(device, commandEncoder, pipelines[103], layouts[103], infinityBuf, [buf_35, buf_18, buf_133, buf_134], [128, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[104], layouts[104], infinityBuf, [buf_18, buf_35, buf_135, buf_136, buf_137, buf_47], [32, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[105], layouts[105], infinityBuf, [buf_14, buf_18], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[106], layouts[106], infinityBuf, [buf_79, buf_18], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[107], layouts[107], infinityBuf, [buf_78, buf_18, buf_14], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[108], layouts[108], infinityBuf, [buf_13, buf_18, buf_79], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[109], layouts[109], infinityBuf, [buf_47, buf_18, buf_14, buf_78, buf_138, buf_139], [48, 580, 1]);
        addComputePass(device, commandEncoder, pipelines[110], layouts[110], infinityBuf, [buf_140, buf_18, buf_79, buf_13, buf_83, buf_84], [24, 2, 12]);
        addComputePass(device, commandEncoder, pipelines[111], layouts[111], infinityBuf, [buf_15, buf_47, buf_141, buf_142], [32, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[112], layouts[112], infinityBuf, [buf_21, buf_47, buf_143, buf_144], [32, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[113], layouts[113], infinityBuf, [buf_24, buf_47, buf_145, buf_146], [32, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[114], layouts[114], infinityBuf, [buf_91, buf_15, buf_21], [20, 116, 6]);
        addComputePass(device, commandEncoder, pipelines[115], layouts[115], infinityBuf, [buf_28, buf_91], [870, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[116], layouts[116], infinityBuf, [buf_29, buf_91, buf_28], [435, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[117], layouts[117], infinityBuf, [buf_21, buf_91, buf_28, buf_29, buf_24], [4, 3, 116]);
        addComputePass(device, commandEncoder, pipelines[118], layouts[118], infinityBuf, [buf_24, buf_21, buf_147, buf_148, buf_149, buf_18], [32, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[119], layouts[119], infinityBuf, [buf_13, buf_24], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[120], layouts[120], infinityBuf, [buf_79, buf_24, buf_13], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[121], layouts[121], infinityBuf, [buf_21, buf_24, buf_13, buf_79, buf_150, buf_151], [24, 290, 1]);
        addComputePass(device, commandEncoder, pipelines[122], layouts[122], infinityBuf, [buf_35, buf_21, buf_152, buf_153], [128, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[123], layouts[123], infinityBuf, [buf_21, buf_35, buf_154, buf_155, buf_156, buf_24], [32, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[124], layouts[124], infinityBuf, [buf_79, buf_21], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[125], layouts[125], infinityBuf, [buf_13, buf_21, buf_79], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[126], layouts[126], infinityBuf, [buf_24, buf_21, buf_79, buf_13, buf_157, buf_158], [24, 290, 1]);
        addComputePass(device, commandEncoder, pipelines[127], layouts[127], infinityBuf, [buf_18, buf_24, buf_159, buf_160], [48, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[128], layouts[128], infinityBuf, [buf_15, buf_24, buf_161, buf_162], [48, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[129], layouts[129], infinityBuf, [buf_47, buf_24, buf_163, buf_164], [48, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[130], layouts[130], infinityBuf, [buf_27, buf_18, buf_15], [145, 6, 4]);
        addComputePass(device, commandEncoder, pipelines[131], layouts[131], infinityBuf, [buf_29, buf_27], [120, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[132], layouts[132], infinityBuf, [buf_28, buf_27, buf_29], [145, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[133], layouts[133], infinityBuf, [buf_15, buf_27, buf_29, buf_28, buf_47], [145, 4, 1]);
        addComputePass(device, commandEncoder, pipelines[134], layouts[134], infinityBuf, [buf_47, buf_15, buf_165, buf_166, buf_167, buf_21], [32, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[135], layouts[135], infinityBuf, [buf_13, buf_47], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[136], layouts[136], infinityBuf, [buf_79, buf_47, buf_13], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[137], layouts[137], infinityBuf, [buf_15, buf_47, buf_13, buf_79, buf_168, buf_169], [24, 290, 1]);
        addComputePass(device, commandEncoder, pipelines[138], layouts[138], infinityBuf, [buf_35, buf_15, buf_170, buf_171], [128, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[139], layouts[139], infinityBuf, [buf_15, buf_35, buf_172, buf_173, buf_174, buf_47], [32, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[140], layouts[140], infinityBuf, [buf_79, buf_15], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[141], layouts[141], infinityBuf, [buf_13, buf_15, buf_79], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[142], layouts[142], infinityBuf, [buf_47, buf_15, buf_79, buf_13, buf_175, buf_176], [24, 290, 1]);
        addComputePass(device, commandEncoder, pipelines[143], layouts[143], infinityBuf, [buf_21, buf_47, buf_177, buf_178], [48, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[144], layouts[144], infinityBuf, [buf_18, buf_47, buf_179, buf_180], [48, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[145], layouts[145], infinityBuf, [buf_24, buf_47, buf_181, buf_182], [48, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[146], layouts[146], infinityBuf, [buf_27, buf_21, buf_18], [145, 6, 4]);
        addComputePass(device, commandEncoder, pipelines[147], layouts[147], infinityBuf, [buf_28, buf_27], [120, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[148], layouts[148], infinityBuf, [buf_29, buf_27, buf_28], [145, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[149], layouts[149], infinityBuf, [buf_18, buf_27, buf_28, buf_29, buf_24], [145, 4, 1]);
        addComputePass(device, commandEncoder, pipelines[150], layouts[150], infinityBuf, [buf_24, buf_18, buf_183, buf_184, buf_185, buf_15], [32, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[151], layouts[151], infinityBuf, [buf_13, buf_24], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[152], layouts[152], infinityBuf, [buf_79, buf_24, buf_13], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[153], layouts[153], infinityBuf, [buf_18, buf_24, buf_13, buf_79, buf_186, buf_187], [24, 290, 1]);
        addComputePass(device, commandEncoder, pipelines[154], layouts[154], infinityBuf, [buf_35, buf_18, buf_188, buf_189], [128, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[155], layouts[155], infinityBuf, [buf_18, buf_35, buf_190, buf_191, buf_192, buf_24], [32, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[156], layouts[156], infinityBuf, [buf_79, buf_18], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[157], layouts[157], infinityBuf, [buf_13, buf_18], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[158], layouts[158], infinityBuf, [buf_78, buf_18, buf_79], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[159], layouts[159], infinityBuf, [buf_14, buf_18, buf_13], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[160], layouts[160], infinityBuf, [buf_24, buf_18, buf_79, buf_78, buf_193, buf_194], [48, 580, 1]);
        addComputePass(device, commandEncoder, pipelines[161], layouts[161], infinityBuf, [buf_195, buf_18, buf_13, buf_14, buf_83, buf_84], [24, 2, 12]);
        addComputePass(device, commandEncoder, pipelines[162], layouts[162], infinityBuf, [buf_15, buf_24, buf_196, buf_197], [32, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[163], layouts[163], infinityBuf, [buf_21, buf_24, buf_198, buf_199], [32, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[164], layouts[164], infinityBuf, [buf_47, buf_24, buf_200, buf_201], [32, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[165], layouts[165], infinityBuf, [buf_91, buf_15, buf_21], [20, 116, 6]);
        addComputePass(device, commandEncoder, pipelines[166], layouts[166], infinityBuf, [buf_29, buf_91], [870, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[167], layouts[167], infinityBuf, [buf_28, buf_91, buf_29], [435, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[168], layouts[168], infinityBuf, [buf_21, buf_91, buf_29, buf_28, buf_47], [4, 3, 116]);
        addComputePass(device, commandEncoder, pipelines[169], layouts[169], infinityBuf, [buf_47, buf_21, buf_202, buf_203, buf_204, buf_18], [32, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[170], layouts[170], infinityBuf, [buf_14, buf_47], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[171], layouts[171], infinityBuf, [buf_13, buf_47, buf_14], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[172], layouts[172], infinityBuf, [buf_21, buf_47, buf_14, buf_13, buf_205, buf_206], [24, 290, 1]);
        addComputePass(device, commandEncoder, pipelines[173], layouts[173], infinityBuf, [buf_35, buf_21, buf_207, buf_208], [128, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[174], layouts[174], infinityBuf, [buf_21, buf_35, buf_209, buf_210, buf_211, buf_47], [32, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[175], layouts[175], infinityBuf, [buf_13, buf_21], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[176], layouts[176], infinityBuf, [buf_14, buf_21, buf_13], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[177], layouts[177], infinityBuf, [buf_47, buf_21, buf_13, buf_14, buf_212, buf_213], [24, 290, 1]);
        addComputePass(device, commandEncoder, pipelines[178], layouts[178], infinityBuf, [buf_18, buf_47, buf_214, buf_215], [48, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[179], layouts[179], infinityBuf, [buf_15, buf_47, buf_216, buf_217], [48, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[180], layouts[180], infinityBuf, [buf_24, buf_47, buf_218, buf_219], [48, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[181], layouts[181], infinityBuf, [buf_27, buf_18, buf_15], [145, 6, 4]);
        addComputePass(device, commandEncoder, pipelines[182], layouts[182], infinityBuf, [buf_28, buf_27], [120, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[183], layouts[183], infinityBuf, [buf_29, buf_27, buf_28], [145, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[184], layouts[184], infinityBuf, [buf_15, buf_27, buf_28, buf_29, buf_24], [145, 4, 1]);
        addComputePass(device, commandEncoder, pipelines[185], layouts[185], infinityBuf, [buf_24, buf_15, buf_220, buf_221, buf_222, buf_21], [32, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[186], layouts[186], infinityBuf, [buf_14, buf_24], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[187], layouts[187], infinityBuf, [buf_13, buf_24, buf_14], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[188], layouts[188], infinityBuf, [buf_15, buf_24, buf_14, buf_13, buf_223, buf_224], [24, 290, 1]);
        addComputePass(device, commandEncoder, pipelines[189], layouts[189], infinityBuf, [buf_35, buf_15, buf_225, buf_226], [128, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[190], layouts[190], infinityBuf, [buf_15, buf_35, buf_227, buf_228, buf_229, buf_24], [32, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[191], layouts[191], infinityBuf, [buf_13, buf_15], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[192], layouts[192], infinityBuf, [buf_14, buf_15, buf_13], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[193], layouts[193], infinityBuf, [buf_24, buf_15, buf_13, buf_14, buf_230, buf_231], [24, 290, 1]);
        addComputePass(device, commandEncoder, pipelines[194], layouts[194], infinityBuf, [buf_21, buf_24, buf_232, buf_233], [48, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[195], layouts[195], infinityBuf, [buf_18, buf_24, buf_234, buf_235], [48, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[196], layouts[196], infinityBuf, [buf_47, buf_24, buf_236, buf_237], [48, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[197], layouts[197], infinityBuf, [buf_27, buf_21, buf_18], [145, 6, 4]);
        addComputePass(device, commandEncoder, pipelines[198], layouts[198], infinityBuf, [buf_29, buf_27], [120, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[199], layouts[199], infinityBuf, [buf_28, buf_27, buf_29], [145, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[200], layouts[200], infinityBuf, [buf_18, buf_27, buf_29, buf_28, buf_47], [145, 4, 1]);
        addComputePass(device, commandEncoder, pipelines[201], layouts[201], infinityBuf, [buf_47, buf_18, buf_238, buf_239, buf_240, buf_15], [32, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[202], layouts[202], infinityBuf, [buf_14, buf_47], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[203], layouts[203], infinityBuf, [buf_13, buf_47, buf_14], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[204], layouts[204], infinityBuf, [buf_18, buf_47, buf_14, buf_13, buf_241, buf_242], [24, 290, 1]);
        addComputePass(device, commandEncoder, pipelines[205], layouts[205], infinityBuf, [buf_35, buf_18, buf_243, buf_244], [128, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[206], layouts[206], infinityBuf, [buf_18, buf_35, buf_245, buf_246, buf_247, buf_47], [32, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[207], layouts[207], infinityBuf, [buf_13, buf_18], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[208], layouts[208], infinityBuf, [buf_14, buf_18, buf_13], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[209], layouts[209], infinityBuf, [buf_248, buf_18, buf_13, buf_14, buf_83, buf_84], [24, 2, 12]);
        addComputePass(device, commandEncoder, pipelines[210], layouts[210], infinityBuf, [buf_249, buf_82, buf_140, buf_195, buf_248], [9216, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[211], layouts[211], infinityBuf, [buf_250, buf_249, buf_251], [64, 6, 1]);
        addComputePass(device, commandEncoder, pipelines[212], layouts[212], infinityBuf, [buf_252, buf_250], [576, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[213], layouts[213], infinityBuf, [buf_253, buf_250, buf_252], [32, 576, 1]);
        addComputePass(device, commandEncoder, pipelines[214], layouts[214], infinityBuf, [buf_252, buf_253], [576, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[215], layouts[215], infinityBuf, [buf_250, buf_253, buf_252, buf_254, buf_255], [192, 32, 1]);
        addComputePass(device, commandEncoder, pipelines[216], layouts[216], infinityBuf, [buf_256, buf_250, buf_257], [8, 24, 6]);
        addComputePass(device, commandEncoder, pipelines[217], layouts[217], infinityBuf, [buf_252, buf_256], [144, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[218], layouts[218], infinityBuf, [buf_258, buf_256, buf_252], [128, 72, 1]);
        addComputePass(device, commandEncoder, pipelines[219], layouts[219], infinityBuf, [buf_252, buf_258], [576, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[220], layouts[220], infinityBuf, [buf_256, buf_258, buf_252, buf_259, buf_260], [576, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[221], layouts[221], infinityBuf, [buf_258, buf_256, buf_261], [32, 24, 24]);
        addComputePass(device, commandEncoder, pipelines[222], layouts[222], infinityBuf, [buf_252, buf_258], [144, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[223], layouts[223], infinityBuf, [buf_256, buf_258, buf_252], [128, 72, 1]);
        addComputePass(device, commandEncoder, pipelines[224], layouts[224], infinityBuf, [buf_252, buf_256], [576, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[225], layouts[225], infinityBuf, [buf_258, buf_256, buf_252, buf_262, buf_263], [576, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[226], layouts[226], infinityBuf, [buf_256, buf_258, buf_264], [32, 24, 24]);
        addComputePass(device, commandEncoder, pipelines[227], layouts[227], infinityBuf, [buf_252, buf_256], [144, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[228], layouts[228], infinityBuf, [buf_265, buf_256, buf_252], [128, 72, 1]);
        addComputePass(device, commandEncoder, pipelines[229], layouts[229], infinityBuf, [buf_252, buf_265], [576, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[230], layouts[230], infinityBuf, [buf_256, buf_265, buf_252, buf_266, buf_267], [576, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[231], layouts[231], infinityBuf, [buf_265, buf_256, buf_268], [32, 24, 24]);
        addComputePass(device, commandEncoder, pipelines[232], layouts[232], infinityBuf, [buf_252, buf_265], [144, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[233], layouts[233], infinityBuf, [buf_256, buf_265, buf_252], [128, 72, 1]);
        addComputePass(device, commandEncoder, pipelines[234], layouts[234], infinityBuf, [buf_252, buf_256], [576, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[235], layouts[235], infinityBuf, [buf_265, buf_256, buf_252, buf_269, buf_270], [576, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[236], layouts[236], infinityBuf, [buf_256, buf_265, buf_271], [32, 24, 24]);
        addComputePass(device, commandEncoder, pipelines[237], layouts[237], infinityBuf, [buf_252, buf_256], [144, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[238], layouts[238], infinityBuf, [buf_272, buf_256, buf_252], [128, 72, 1]);
        addComputePass(device, commandEncoder, pipelines[239], layouts[239], infinityBuf, [buf_252, buf_272], [576, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[240], layouts[240], infinityBuf, [buf_256, buf_272, buf_252, buf_273, buf_274], [576, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[241], layouts[241], infinityBuf, [buf_272, buf_256, buf_275], [32, 24, 24]);
        addComputePass(device, commandEncoder, pipelines[242], layouts[242], infinityBuf, [buf_252, buf_272], [144, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[243], layouts[243], infinityBuf, [buf_256, buf_272, buf_252], [128, 72, 1]);
        addComputePass(device, commandEncoder, pipelines[244], layouts[244], infinityBuf, [buf_252, buf_256], [576, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[245], layouts[245], infinityBuf, [buf_276, buf_250, buf_258, buf_265, buf_256, buf_252, buf_277, buf_278], [36, 160, 1]);
        addComputePass(device, commandEncoder, pipelines[246], layouts[246], infinityBuf, [buf_250, buf_276, buf_279], [64, 6, 1]);
        addComputePass(device, commandEncoder, pipelines[247], layouts[247], infinityBuf, [buf_252, buf_250], [576, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[248], layouts[248], infinityBuf, [buf_253, buf_250, buf_252], [32, 576, 1]);
        addComputePass(device, commandEncoder, pipelines[249], layouts[249], infinityBuf, [buf_252, buf_253], [576, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[250], layouts[250], infinityBuf, [buf_250, buf_253, buf_252, buf_280, buf_281], [16, 576, 1]);
        addComputePass(device, commandEncoder, pipelines[251], layouts[251], infinityBuf, [buf_252, buf_250], [576, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[252], layouts[252], infinityBuf, [buf_253, buf_250, buf_252], [32, 576, 1]);
        addComputePass(device, commandEncoder, pipelines[253], layouts[253], infinityBuf, [buf_252, buf_253], [576, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[254], layouts[254], infinityBuf, [buf_250, buf_253, buf_252, buf_282, buf_283], [16, 576, 1]);
        addComputePass(device, commandEncoder, pipelines[255], layouts[255], infinityBuf, [buf_253, buf_250, buf_284, buf_285], [24, 8, 8]);
        addComputePass(device, commandEncoder, pipelines[256], layouts[256], infinityBuf, [buf_286, buf_250, buf_287, buf_288], [2, 24, 1]);
        addComputePass(device, commandEncoder, pipelines[257], layouts[257], infinityBuf, [buf_289, buf_250, buf_290, buf_291], [24, 8, 8]);
        addComputePass(device, commandEncoder, pipelines[258], layouts[258], infinityBuf, [buf_252, buf_286], [576, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[259], layouts[259], infinityBuf, [buf_292, buf_286, buf_252], [576, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[260], layouts[260], infinityBuf, [buf_250, buf_286, buf_252, buf_292, buf_293, buf_294], [64, 72, 1]);
        addComputePass(device, commandEncoder, pipelines[261], layouts[261], infinityBuf, [buf_292, buf_250, buf_295, buf_296], [48, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[262], layouts[262], infinityBuf, [buf_286, buf_250, buf_297, buf_298], [16, 144, 1]);
        addComputePass(device, commandEncoder, pipelines[263], layouts[263], infinityBuf, [buf_299, buf_292], [2, 256, 1]);
        addComputePass(device, commandEncoder, pipelines[264], layouts[264], infinityBuf, [buf_300, buf_292], [576, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[265], layouts[265], infinityBuf, [buf_250, buf_286, buf_301, buf_302], [16, 144, 1]);
        addComputePass(device, commandEncoder, pipelines[266], layouts[266], infinityBuf, [buf_303, buf_299], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[267], layouts[267], infinityBuf, [buf_304, buf_250, buf_305, buf_306], [4, 576, 1]);
        addComputePass(device, commandEncoder, pipelines[268], layouts[268], infinityBuf, [buf_299, buf_303], [2, 2, 128]);
        addComputePass(device, commandEncoder, pipelines[269], layouts[269], infinityBuf, [buf_303, buf_299], [2, 256, 1]);
        addComputePass(device, commandEncoder, pipelines[270], layouts[270], infinityBuf, [buf_299, buf_303], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[271], layouts[271], infinityBuf, [buf_303, buf_299], [4, 2, 64]);
        addComputePass(device, commandEncoder, pipelines[272], layouts[272], infinityBuf, [buf_299, buf_303], [4, 128, 1]);
        addComputePass(device, commandEncoder, pipelines[273], layouts[273], infinityBuf, [buf_303, buf_299], [2, 256, 1]);
        addComputePass(device, commandEncoder, pipelines[274], layouts[274], infinityBuf, [buf_299, buf_303], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[275], layouts[275], infinityBuf, [buf_303, buf_299], [8, 2, 32]);
        addComputePass(device, commandEncoder, pipelines[276], layouts[276], infinityBuf, [buf_299, buf_303], [8, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[277], layouts[277], infinityBuf, [buf_303, buf_299], [4, 128, 1]);
        addComputePass(device, commandEncoder, pipelines[278], layouts[278], infinityBuf, [buf_299, buf_303], [2, 256, 1]);
        addComputePass(device, commandEncoder, pipelines[279], layouts[279], infinityBuf, [buf_303, buf_299], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[280], layouts[280], infinityBuf, [buf_299, buf_303], [16, 2, 2]);
        addComputePass(device, commandEncoder, pipelines[281], layouts[281], infinityBuf, [buf_303, buf_299], [16, 32, 1]);
        addComputePass(device, commandEncoder, pipelines[282], layouts[282], infinityBuf, [buf_299, buf_303], [8, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[283], layouts[283], infinityBuf, [buf_303, buf_299], [4, 128, 1]);
        addComputePass(device, commandEncoder, pipelines[284], layouts[284], infinityBuf, [buf_299, buf_303], [2, 256, 1]);
        addComputePass(device, commandEncoder, pipelines[285], layouts[285], infinityBuf, [buf_303, buf_299], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[286], layouts[286], infinityBuf, [buf_299, buf_303], [32, 2, 2]);
        addComputePass(device, commandEncoder, pipelines[287], layouts[287], infinityBuf, [buf_303, buf_299], [32, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[288], layouts[288], infinityBuf, [buf_299, buf_303], [16, 32, 1]);
        addComputePass(device, commandEncoder, pipelines[289], layouts[289], infinityBuf, [buf_303, buf_299], [8, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[290], layouts[290], infinityBuf, [buf_299, buf_303], [4, 128, 1]);
        addComputePass(device, commandEncoder, pipelines[291], layouts[291], infinityBuf, [buf_303, buf_299], [2, 256, 1]);
        addComputePass(device, commandEncoder, pipelines[292], layouts[292], infinityBuf, [buf_299, buf_303], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[293], layouts[293], infinityBuf, [buf_303, buf_299], [64, 2, 2]);
        addComputePass(device, commandEncoder, pipelines[294], layouts[294], infinityBuf, [buf_299, buf_303], [64, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[295], layouts[295], infinityBuf, [buf_303, buf_299], [32, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[296], layouts[296], infinityBuf, [buf_299, buf_303], [16, 32, 1]);
        addComputePass(device, commandEncoder, pipelines[297], layouts[297], infinityBuf, [buf_303, buf_299], [8, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[298], layouts[298], infinityBuf, [buf_299, buf_303], [4, 128, 1]);
        addComputePass(device, commandEncoder, pipelines[299], layouts[299], infinityBuf, [buf_303, buf_299], [2, 256, 1]);
        addComputePass(device, commandEncoder, pipelines[300], layouts[300], infinityBuf, [buf_299, buf_303], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[301], layouts[301], infinityBuf, [buf_303, buf_299], [128, 2, 2]);
        addComputePass(device, commandEncoder, pipelines[302], layouts[302], infinityBuf, [buf_299, buf_303], [128, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[303], layouts[303], infinityBuf, [buf_303, buf_299], [64, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[304], layouts[304], infinityBuf, [buf_299, buf_303], [32, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[305], layouts[305], infinityBuf, [buf_303, buf_299], [16, 32, 1]);
        addComputePass(device, commandEncoder, pipelines[306], layouts[306], infinityBuf, [buf_299, buf_303], [8, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[307], layouts[307], infinityBuf, [buf_303, buf_299], [4, 128, 1]);
        addComputePass(device, commandEncoder, pipelines[308], layouts[308], infinityBuf, [buf_299, buf_303], [2, 256, 1]);
        addComputePass(device, commandEncoder, pipelines[309], layouts[309], infinityBuf, [buf_303, buf_299], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[310], layouts[310], infinityBuf, [buf_299, buf_303], [256, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[311], layouts[311], infinityBuf, [buf_303, buf_299], [256, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[312], layouts[312], infinityBuf, [buf_299, buf_303], [128, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[313], layouts[313], infinityBuf, [buf_303, buf_299], [64, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[314], layouts[314], infinityBuf, [buf_299, buf_303], [32, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[315], layouts[315], infinityBuf, [buf_303, buf_299], [16, 32, 1]);
        addComputePass(device, commandEncoder, pipelines[316], layouts[316], infinityBuf, [buf_299, buf_303], [8, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[317], layouts[317], infinityBuf, [buf_303, buf_299], [4, 128, 1]);
        addComputePass(device, commandEncoder, pipelines[318], layouts[318], infinityBuf, [buf_299, buf_303], [2, 256, 1]);
        addComputePass(device, commandEncoder, pipelines[319], layouts[319], infinityBuf, [buf_303, buf_299], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[320], layouts[320], infinityBuf, [buf_299, buf_303], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[321], layouts[321], infinityBuf, [buf_303, buf_299], [256, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[322], layouts[322], infinityBuf, [buf_299, buf_303], [128, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[323], layouts[323], infinityBuf, [buf_303, buf_299], [64, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[324], layouts[324], infinityBuf, [buf_299, buf_303], [32, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[325], layouts[325], infinityBuf, [buf_303, buf_299], [16, 32, 1]);
        addComputePass(device, commandEncoder, pipelines[326], layouts[326], infinityBuf, [buf_299, buf_303], [8, 64, 1]);
        addComputePass(device, commandEncoder, pipelines[327], layouts[327], infinityBuf, [buf_303, buf_299], [4, 128, 1]);
        addComputePass(device, commandEncoder, pipelines[328], layouts[328], infinityBuf, [buf_299, buf_303], [2, 256, 1]);
        addComputePass(device, commandEncoder, pipelines[329], layouts[329], infinityBuf, [buf_303, buf_299], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[330], layouts[330], infinityBuf, [buf_307, buf_303], [72, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[331], layouts[331], infinityBuf, [buf_308, buf_292, buf_303, buf_300, buf_307], [144, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[332], layouts[332], infinityBuf, [buf_309, buf_308, buf_304], [150, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[333], layouts[333], infinityBuf, [buf_310, buf_311, buf_309, buf_312, buf_313], [32, 300, 1]);
        addComputePass(device, commandEncoder, pipelines[334], layouts[334], infinityBuf, [buf_314, buf_310, buf_315, buf_316], [4, 15, 1]);
        addComputePass(device, commandEncoder, pipelines[335], layouts[335], infinityBuf, [buf_310, buf_2, buf_314, buf_3, buf_4], [2, 75, 1]);
        addComputePass(device, commandEncoder, pipelines[336], layouts[336], infinityBuf, [buf_317, buf_2, buf_314, buf_3, buf_4], [32, 75, 1]);
        addComputePass(device, commandEncoder, pipelines[337], layouts[337], infinityBuf, [buf_318, buf_310, buf_317], [75, 300, 2]);
        addComputePass(device, commandEncoder, pipelines[338], layouts[338], infinityBuf, [buf_319, buf_318], [600, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[339], layouts[339], infinityBuf, [buf_320, buf_318, buf_319], [150, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[340], layouts[340], infinityBuf, [buf_317, buf_318, buf_319, buf_320, buf_1], [8, 60, 1]);
        addComputePass(device, commandEncoder, pipelines[341], layouts[341], infinityBuf, [buf_1, buf_2, buf_317, buf_321, buf_322], [16, 300, 1]);
        addComputePass(device, commandEncoder, pipelines[342], layouts[342], infinityBuf, [buf_323, buf_1], [150, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[343], layouts[343], infinityBuf, [buf_324, buf_1, buf_323], [300, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[344], layouts[344], infinityBuf, [buf_317, buf_1, buf_323, buf_324, buf_325, buf_326], [16, 300, 1]);
        addComputePass(device, commandEncoder, pipelines[345], layouts[345], infinityBuf, [buf_327, buf_311, buf_309, buf_317, buf_314, buf_328, buf_329], [2, 2, 150]);
        addComputePass(device, commandEncoder, pipelines[346], layouts[346], infinityBuf, [buf_330, buf_317, buf_314, buf_331, buf_332], [32, 75, 1]);
        addComputePass(device, commandEncoder, pipelines[347], layouts[347], infinityBuf, [buf_333, buf_330], [2400, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[348], layouts[348], infinityBuf, [buf_334, buf_330, buf_333], [2400, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[349], layouts[349], infinityBuf, [buf_1, buf_327, buf_253, buf_330, buf_333, buf_334], [16, 150, 1]);
        addComputePass(device, commandEncoder, pipelines[350], layouts[350], infinityBuf, [buf_310, buf_317, buf_1, buf_335, buf_336], [75, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[351], layouts[351], infinityBuf, [buf_324, buf_310], [150, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[352], layouts[352], infinityBuf, [buf_323, buf_310, buf_324], [300, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[353], layouts[353], infinityBuf, [buf_1, buf_310, buf_324, buf_323, buf_337, buf_338], [16, 300, 1]);
        addComputePass(device, commandEncoder, pipelines[354], layouts[354], infinityBuf, [buf_339, buf_1, buf_340, buf_341], [64, 15, 1]);
        addComputePass(device, commandEncoder, pipelines[355], layouts[355], infinityBuf, [buf_310, buf_1, buf_339, buf_342, buf_343], [8, 60, 1]);
        addComputePass(device, commandEncoder, pipelines[356], layouts[356], infinityBuf, [buf_323, buf_310], [150, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[357], layouts[357], infinityBuf, [buf_324, buf_310, buf_323], [300, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[358], layouts[358], infinityBuf, [buf_1, buf_310, buf_323, buf_324, buf_344, buf_345], [16, 300, 1]);
        addComputePass(device, commandEncoder, pipelines[359], layouts[359], infinityBuf, [buf_310, buf_1, buf_314, buf_346, buf_347], [16, 60, 1]);
        addComputePass(device, commandEncoder, pipelines[360], layouts[360], infinityBuf, [buf_317, buf_1, buf_314, buf_346, buf_347], [16, 60, 1]);
        addComputePass(device, commandEncoder, pipelines[361], layouts[361], infinityBuf, [buf_348, buf_1, buf_346, buf_347], [16, 30, 1]);
        addComputePass(device, commandEncoder, pipelines[362], layouts[362], infinityBuf, [buf_324, buf_1], [150, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[363], layouts[363], infinityBuf, [buf_318, buf_310, buf_317], [75, 300, 2]);
        addComputePass(device, commandEncoder, pipelines[364], layouts[364], infinityBuf, [buf_323, buf_1, buf_324], [300, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[365], layouts[365], infinityBuf, [buf_320, buf_318], [600, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[366], layouts[366], infinityBuf, [buf_349, buf_1, buf_324, buf_323, buf_350, buf_351], [16, 300, 2]);
        addComputePass(device, commandEncoder, pipelines[367], layouts[367], infinityBuf, [buf_319, buf_318, buf_320], [150, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[368], layouts[368], infinityBuf, [buf_317, buf_318, buf_320, buf_319, buf_348], [8, 60, 1]);
        addComputePass(device, commandEncoder, pipelines[369], layouts[369], infinityBuf, [buf_348, buf_1, buf_317, buf_352, buf_353], [75, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[370], layouts[370], infinityBuf, [buf_323, buf_348], [150, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[371], layouts[371], infinityBuf, [buf_324, buf_348, buf_323], [300, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[372], layouts[372], infinityBuf, [buf_317, buf_348, buf_323, buf_324, buf_354, buf_355], [16, 300, 1]);
        addComputePass(device, commandEncoder, pipelines[373], layouts[373], infinityBuf, [buf_327, buf_311, buf_309, buf_317, buf_314, buf_356, buf_357], [2, 2, 150]);
        addComputePass(device, commandEncoder, pipelines[374], layouts[374], infinityBuf, [buf_330, buf_317, buf_314, buf_358, buf_359], [32, 75, 1]);
        addComputePass(device, commandEncoder, pipelines[375], layouts[375], infinityBuf, [buf_334, buf_330], [2400, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[376], layouts[376], infinityBuf, [buf_333, buf_330, buf_334], [2400, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[377], layouts[377], infinityBuf, [buf_314, buf_327, buf_289, buf_330, buf_334, buf_333], [16, 150, 1]);
        addComputePass(device, commandEncoder, pipelines[378], layouts[378], infinityBuf, [buf_348, buf_317, buf_314, buf_360, buf_361], [75, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[379], layouts[379], infinityBuf, [buf_324, buf_348], [150, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[380], layouts[380], infinityBuf, [buf_323, buf_348, buf_324], [300, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[381], layouts[381], infinityBuf, [buf_314, buf_348, buf_324, buf_323, buf_362, buf_363], [16, 300, 1]);
        addComputePass(device, commandEncoder, pipelines[382], layouts[382], infinityBuf, [buf_339, buf_314, buf_364, buf_365], [64, 15, 1]);
        addComputePass(device, commandEncoder, pipelines[383], layouts[383], infinityBuf, [buf_348, buf_314, buf_339, buf_366, buf_367], [8, 60, 1]);
        addComputePass(device, commandEncoder, pipelines[384], layouts[384], infinityBuf, [buf_323, buf_348], [150, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[385], layouts[385], infinityBuf, [buf_324, buf_348, buf_323], [300, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[386], layouts[386], infinityBuf, [buf_314, buf_348, buf_323, buf_324, buf_368, buf_369], [16, 300, 1]);
        addComputePass(device, commandEncoder, pipelines[387], layouts[387], infinityBuf, [buf_324, buf_314], [150, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[388], layouts[388], infinityBuf, [buf_323, buf_314, buf_324], [300, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[389], layouts[389], infinityBuf, [buf_370, buf_314, buf_324, buf_323, buf_350, buf_351], [16, 300, 2]);
        addComputePass(device, commandEncoder, pipelines[390], layouts[390], infinityBuf, [buf_371, buf_349, buf_370], [4800, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[391], layouts[391], infinityBuf, [buf_372, buf_371, buf_373, buf_374], [300, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[392], layouts[392], infinityBuf, [buf_370, buf_371, buf_375, buf_376], [200, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[393], layouts[393], infinityBuf, [buf_377, buf_372], [512, 2, 4]);
        addComputePass(device, commandEncoder, pipelines[394], layouts[394], infinityBuf, [buf_378, buf_372], [1950, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[395], layouts[395], infinityBuf, [buf_371, buf_370, buf_379, buf_380], [200, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[396], layouts[396], infinityBuf, [buf_381, buf_377], [2, 1024, 1]);
        addComputePass(device, commandEncoder, pipelines[397], layouts[397], infinityBuf, [buf_319, buf_371, buf_382, buf_383], [4, 75, 1]);
        addComputePass(device, commandEncoder, pipelines[398], layouts[398], infinityBuf, [buf_377, buf_381], [4, 2, 512]);
        addComputePass(device, commandEncoder, pipelines[399], layouts[399], infinityBuf, [buf_381, buf_377], [2, 2, 2048]);
        addComputePass(device, commandEncoder, pipelines[400], layouts[400], infinityBuf, [buf_377, buf_381], [2, 1024, 1]);
        addComputePass(device, commandEncoder, pipelines[401], layouts[401], infinityBuf, [buf_381, buf_377], [2, 2, 2048]);
        addComputePass(device, commandEncoder, pipelines[402], layouts[402], infinityBuf, [buf_377, buf_381], [4, 2, 1024]);
        addComputePass(device, commandEncoder, pipelines[403], layouts[403], infinityBuf, [buf_381, buf_377], [2, 2, 2048]);
        addComputePass(device, commandEncoder, pipelines[404], layouts[404], infinityBuf, [buf_377, buf_381], [2, 1024, 1]);
        addComputePass(device, commandEncoder, pipelines[405], layouts[405], infinityBuf, [buf_381, buf_377], [4, 2, 1024]);
        addComputePass(device, commandEncoder, pipelines[406], layouts[406], infinityBuf, [buf_377, buf_381], [2, 2048, 1]);
        addComputePass(device, commandEncoder, pipelines[407], layouts[407], infinityBuf, [buf_381, buf_377], [4, 2, 1024]);
        addComputePass(device, commandEncoder, pipelines[408], layouts[408], infinityBuf, [buf_377, buf_381], [2, 2, 2048]);
        addComputePass(device, commandEncoder, pipelines[409], layouts[409], infinityBuf, [buf_381, buf_377], [2, 1024, 1]);
        addComputePass(device, commandEncoder, pipelines[410], layouts[410], infinityBuf, [buf_377, buf_381], [16, 2, 512]);
        addComputePass(device, commandEncoder, pipelines[411], layouts[411], infinityBuf, [buf_381, buf_377], [4, 2, 1024]);
        addComputePass(device, commandEncoder, pipelines[412], layouts[412], infinityBuf, [buf_377, buf_381], [2, 2048, 1]);
        addComputePass(device, commandEncoder, pipelines[413], layouts[413], infinityBuf, [buf_381, buf_377], [4, 2, 1024]);
        addComputePass(device, commandEncoder, pipelines[414], layouts[414], infinityBuf, [buf_377, buf_381], [2, 2, 2048]);
        addComputePass(device, commandEncoder, pipelines[415], layouts[415], infinityBuf, [buf_381, buf_377], [2, 1024, 1]);
        addComputePass(device, commandEncoder, pipelines[416], layouts[416], infinityBuf, [buf_377, buf_381], [64, 2, 32]);
        addComputePass(device, commandEncoder, pipelines[417], layouts[417], infinityBuf, [buf_381, buf_377], [32, 2, 128]);
        addComputePass(device, commandEncoder, pipelines[418], layouts[418], infinityBuf, [buf_377, buf_381], [4, 2, 1024]);
        addComputePass(device, commandEncoder, pipelines[419], layouts[419], infinityBuf, [buf_381, buf_377], [2, 2048, 1]);
        addComputePass(device, commandEncoder, pipelines[420], layouts[420], infinityBuf, [buf_377, buf_381], [4, 2, 1024]);
        addComputePass(device, commandEncoder, pipelines[421], layouts[421], infinityBuf, [buf_381, buf_377], [2, 2, 2048]);
        addComputePass(device, commandEncoder, pipelines[422], layouts[422], infinityBuf, [buf_377, buf_381], [2, 1024, 1]);
        addComputePass(device, commandEncoder, pipelines[423], layouts[423], infinityBuf, [buf_381, buf_377], [128, 2, 16]);
        addComputePass(device, commandEncoder, pipelines[424], layouts[424], infinityBuf, [buf_377, buf_381], [16, 2, 256]);
        addComputePass(device, commandEncoder, pipelines[425], layouts[425], infinityBuf, [buf_381, buf_377], [32, 2, 128]);
        addComputePass(device, commandEncoder, pipelines[426], layouts[426], infinityBuf, [buf_377, buf_381], [4, 2, 1024]);
        addComputePass(device, commandEncoder, pipelines[427], layouts[427], infinityBuf, [buf_381, buf_377], [2, 2048, 1]);
        addComputePass(device, commandEncoder, pipelines[428], layouts[428], infinityBuf, [buf_377, buf_381], [4, 2, 1024]);
        addComputePass(device, commandEncoder, pipelines[429], layouts[429], infinityBuf, [buf_381, buf_377], [2, 2, 2048]);
        addComputePass(device, commandEncoder, pipelines[430], layouts[430], infinityBuf, [buf_377, buf_381], [2, 1024, 1]);
        addComputePass(device, commandEncoder, pipelines[431], layouts[431], infinityBuf, [buf_381, buf_377], [64, 2, 64]);
        addComputePass(device, commandEncoder, pipelines[432], layouts[432], infinityBuf, [buf_377, buf_381], [32, 2, 128]);
        addComputePass(device, commandEncoder, pipelines[433], layouts[433], infinityBuf, [buf_381, buf_377], [16, 2, 256]);
        addComputePass(device, commandEncoder, pipelines[434], layouts[434], infinityBuf, [buf_377, buf_381], [32, 2, 128]);
        addComputePass(device, commandEncoder, pipelines[435], layouts[435], infinityBuf, [buf_381, buf_377], [4, 2, 1024]);
        addComputePass(device, commandEncoder, pipelines[436], layouts[436], infinityBuf, [buf_377, buf_381], [2, 2048, 1]);
        addComputePass(device, commandEncoder, pipelines[437], layouts[437], infinityBuf, [buf_381, buf_377], [4, 2, 1024]);
        addComputePass(device, commandEncoder, pipelines[438], layouts[438], infinityBuf, [buf_377, buf_381], [2, 2, 2048]);
        addComputePass(device, commandEncoder, pipelines[439], layouts[439], infinityBuf, [buf_381, buf_377], [2, 1024, 1]);
        addComputePass(device, commandEncoder, pipelines[440], layouts[440], infinityBuf, [buf_377, buf_381], [512, 2, 8]);
        addComputePass(device, commandEncoder, pipelines[441], layouts[441], infinityBuf, [buf_381, buf_377], [64, 2, 64]);
        addComputePass(device, commandEncoder, pipelines[442], layouts[442], infinityBuf, [buf_377, buf_381], [32, 2, 128]);
        addComputePass(device, commandEncoder, pipelines[443], layouts[443], infinityBuf, [buf_381, buf_377], [16, 2, 256]);
        addComputePass(device, commandEncoder, pipelines[444], layouts[444], infinityBuf, [buf_377, buf_381], [32, 2, 128]);
        addComputePass(device, commandEncoder, pipelines[445], layouts[445], infinityBuf, [buf_381, buf_377], [4, 2, 1024]);
        addComputePass(device, commandEncoder, pipelines[446], layouts[446], infinityBuf, [buf_377, buf_381], [2, 2048, 1]);
        addComputePass(device, commandEncoder, pipelines[447], layouts[447], infinityBuf, [buf_381, buf_377], [4, 2, 1024]);
        addComputePass(device, commandEncoder, pipelines[448], layouts[448], infinityBuf, [buf_377, buf_381], [2, 2, 2048]);
        addComputePass(device, commandEncoder, pipelines[449], layouts[449], infinityBuf, [buf_381, buf_377], [2, 1024, 1]);
        addComputePass(device, commandEncoder, pipelines[450], layouts[450], infinityBuf, [buf_377, buf_381], [512, 2, 2]);
        addComputePass(device, commandEncoder, pipelines[451], layouts[451], infinityBuf, [buf_381, buf_377], [512, 2, 8]);
        addComputePass(device, commandEncoder, pipelines[452], layouts[452], infinityBuf, [buf_377, buf_381], [64, 2, 64]);
        addComputePass(device, commandEncoder, pipelines[453], layouts[453], infinityBuf, [buf_381, buf_377], [32, 2, 128]);
        addComputePass(device, commandEncoder, pipelines[454], layouts[454], infinityBuf, [buf_377, buf_381], [16, 2, 256]);
        addComputePass(device, commandEncoder, pipelines[455], layouts[455], infinityBuf, [buf_381, buf_377], [32, 2, 128]);
        addComputePass(device, commandEncoder, pipelines[456], layouts[456], infinityBuf, [buf_377, buf_381], [4, 2, 1024]);
        addComputePass(device, commandEncoder, pipelines[457], layouts[457], infinityBuf, [buf_381, buf_377], [2, 2048, 1]);
        addComputePass(device, commandEncoder, pipelines[458], layouts[458], infinityBuf, [buf_377, buf_381], [4, 2, 1024]);
        addComputePass(device, commandEncoder, pipelines[459], layouts[459], infinityBuf, [buf_381, buf_377], [2, 2, 2048]);
        addComputePass(device, commandEncoder, pipelines[460], layouts[460], infinityBuf, [buf_377, buf_381], [2, 1024, 1]);
        addComputePass(device, commandEncoder, pipelines[461], layouts[461], infinityBuf, [buf_381, buf_377], [2048, 2, 2]);
        addComputePass(device, commandEncoder, pipelines[462], layouts[462], infinityBuf, [buf_377, buf_381], [1024, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[463], layouts[463], infinityBuf, [buf_381, buf_377], [512, 2, 8]);
        addComputePass(device, commandEncoder, pipelines[464], layouts[464], infinityBuf, [buf_377, buf_381], [64, 2, 64]);
        addComputePass(device, commandEncoder, pipelines[465], layouts[465], infinityBuf, [buf_381, buf_377], [32, 2, 128]);
        addComputePass(device, commandEncoder, pipelines[466], layouts[466], infinityBuf, [buf_377, buf_381], [16, 2, 256]);
        addComputePass(device, commandEncoder, pipelines[467], layouts[467], infinityBuf, [buf_381, buf_377], [32, 2, 128]);
        addComputePass(device, commandEncoder, pipelines[468], layouts[468], infinityBuf, [buf_377, buf_381], [4, 2, 1024]);
        addComputePass(device, commandEncoder, pipelines[469], layouts[469], infinityBuf, [buf_381, buf_377], [2, 2048, 1]);
        addComputePass(device, commandEncoder, pipelines[470], layouts[470], infinityBuf, [buf_377, buf_381], [4, 2, 1024]);
        addComputePass(device, commandEncoder, pipelines[471], layouts[471], infinityBuf, [buf_381, buf_377], [2, 2, 2048]);
        addComputePass(device, commandEncoder, pipelines[472], layouts[472], infinityBuf, [buf_377, buf_381], [2, 1024, 1]);
        addComputePass(device, commandEncoder, pipelines[473], layouts[473], infinityBuf, [buf_381, buf_377], [2048, 2, 4]);
        addComputePass(device, commandEncoder, pipelines[474], layouts[474], infinityBuf, [buf_377, buf_381], [2048, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[475], layouts[475], infinityBuf, [buf_381, buf_377], [1024, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[476], layouts[476], infinityBuf, [buf_377, buf_381], [512, 2, 8]);
        addComputePass(device, commandEncoder, pipelines[477], layouts[477], infinityBuf, [buf_381, buf_377], [64, 2, 64]);
        addComputePass(device, commandEncoder, pipelines[478], layouts[478], infinityBuf, [buf_377, buf_381], [32, 2, 128]);
        addComputePass(device, commandEncoder, pipelines[479], layouts[479], infinityBuf, [buf_381, buf_377], [16, 2, 256]);
        addComputePass(device, commandEncoder, pipelines[480], layouts[480], infinityBuf, [buf_377, buf_381], [32, 2, 128]);
        addComputePass(device, commandEncoder, pipelines[481], layouts[481], infinityBuf, [buf_381, buf_377], [4, 2, 1024]);
        addComputePass(device, commandEncoder, pipelines[482], layouts[482], infinityBuf, [buf_377, buf_381], [2, 2048, 1]);
        addComputePass(device, commandEncoder, pipelines[483], layouts[483], infinityBuf, [buf_381, buf_377], [4, 2, 1024]);
        addComputePass(device, commandEncoder, pipelines[484], layouts[484], infinityBuf, [buf_377, buf_381], [2, 2, 2048]);
        addComputePass(device, commandEncoder, pipelines[485], layouts[485], infinityBuf, [buf_381, buf_377], [2, 1024, 1]);
        addComputePass(device, commandEncoder, pipelines[486], layouts[486], infinityBuf, [buf_377, buf_381], [2048, 2, 2]);
        addComputePass(device, commandEncoder, pipelines[487], layouts[487], infinityBuf, [buf_381, buf_377], [4096, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[488], layouts[488], infinityBuf, [buf_377, buf_381], [2048, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[489], layouts[489], infinityBuf, [buf_381, buf_377], [1024, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[490], layouts[490], infinityBuf, [buf_377, buf_381], [512, 2, 8]);
        addComputePass(device, commandEncoder, pipelines[491], layouts[491], infinityBuf, [buf_381, buf_377], [64, 2, 64]);
        addComputePass(device, commandEncoder, pipelines[492], layouts[492], infinityBuf, [buf_377, buf_381], [32, 2, 128]);
        addComputePass(device, commandEncoder, pipelines[493], layouts[493], infinityBuf, [buf_381, buf_377], [16, 2, 256]);
        addComputePass(device, commandEncoder, pipelines[494], layouts[494], infinityBuf, [buf_377, buf_381], [32, 2, 128]);
        addComputePass(device, commandEncoder, pipelines[495], layouts[495], infinityBuf, [buf_381, buf_377], [4, 2, 1024]);
        addComputePass(device, commandEncoder, pipelines[496], layouts[496], infinityBuf, [buf_377, buf_381], [2, 2048, 1]);
        addComputePass(device, commandEncoder, pipelines[497], layouts[497], infinityBuf, [buf_381, buf_377], [4, 2, 1024]);
        addComputePass(device, commandEncoder, pipelines[498], layouts[498], infinityBuf, [buf_377, buf_381], [2, 2, 2048]);
        addComputePass(device, commandEncoder, pipelines[499], layouts[499], infinityBuf, [buf_381, buf_377], [2, 1024, 1]);
        addComputePass(device, commandEncoder, pipelines[500], layouts[500], infinityBuf, [buf_377, buf_381], [2048, 2, 2]);
        addComputePass(device, commandEncoder, pipelines[501], layouts[501], infinityBuf, [buf_381, buf_377], [2048, 2, 2]);
        addComputePass(device, commandEncoder, pipelines[502], layouts[502], infinityBuf, [buf_377, buf_381], [4096, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[503], layouts[503], infinityBuf, [buf_381, buf_377], [2048, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[504], layouts[504], infinityBuf, [buf_377, buf_381], [1024, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[505], layouts[505], infinityBuf, [buf_381, buf_377], [512, 2, 8]);
        addComputePass(device, commandEncoder, pipelines[506], layouts[506], infinityBuf, [buf_377, buf_381], [64, 2, 64]);
        addComputePass(device, commandEncoder, pipelines[507], layouts[507], infinityBuf, [buf_381, buf_377], [32, 2, 128]);
        addComputePass(device, commandEncoder, pipelines[508], layouts[508], infinityBuf, [buf_377, buf_381], [16, 2, 256]);
        addComputePass(device, commandEncoder, pipelines[509], layouts[509], infinityBuf, [buf_381, buf_377], [32, 2, 128]);
        addComputePass(device, commandEncoder, pipelines[510], layouts[510], infinityBuf, [buf_377, buf_381], [4, 2, 1024]);
        addComputePass(device, commandEncoder, pipelines[511], layouts[511], infinityBuf, [buf_381, buf_377], [2, 2048, 1]);
        addComputePass(device, commandEncoder, pipelines[512], layouts[512], infinityBuf, [buf_377, buf_381], [4, 2, 1024]);
        addComputePass(device, commandEncoder, pipelines[513], layouts[513], infinityBuf, [buf_381, buf_377], [2, 2, 2048]);
        addComputePass(device, commandEncoder, pipelines[514], layouts[514], infinityBuf, [buf_377, buf_381], [2, 1024, 1]);
        addComputePass(device, commandEncoder, pipelines[515], layouts[515], infinityBuf, [buf_381, buf_377], [4096, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[516], layouts[516], infinityBuf, [buf_377, buf_381], [2048, 2, 2]);
        addComputePass(device, commandEncoder, pipelines[517], layouts[517], infinityBuf, [buf_381, buf_377], [4096, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[518], layouts[518], infinityBuf, [buf_377, buf_381], [2048, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[519], layouts[519], infinityBuf, [buf_381, buf_377], [1024, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[520], layouts[520], infinityBuf, [buf_377, buf_381], [512, 2, 8]);
        addComputePass(device, commandEncoder, pipelines[521], layouts[521], infinityBuf, [buf_381, buf_377], [64, 2, 64]);
        addComputePass(device, commandEncoder, pipelines[522], layouts[522], infinityBuf, [buf_377, buf_381], [32, 2, 128]);
        addComputePass(device, commandEncoder, pipelines[523], layouts[523], infinityBuf, [buf_381, buf_377], [16, 2, 256]);
        addComputePass(device, commandEncoder, pipelines[524], layouts[524], infinityBuf, [buf_377, buf_381], [32, 2, 128]);
        addComputePass(device, commandEncoder, pipelines[525], layouts[525], infinityBuf, [buf_381, buf_377], [4, 2, 1024]);
        addComputePass(device, commandEncoder, pipelines[526], layouts[526], infinityBuf, [buf_377, buf_381], [2, 2048, 1]);
        addComputePass(device, commandEncoder, pipelines[527], layouts[527], infinityBuf, [buf_381, buf_377], [4, 2, 1024]);
        addComputePass(device, commandEncoder, pipelines[528], layouts[528], infinityBuf, [buf_377, buf_381], [2, 2, 2048]);
        addComputePass(device, commandEncoder, pipelines[529], layouts[529], infinityBuf, [buf_381, buf_377], [2, 1024, 1]);
        addComputePass(device, commandEncoder, pipelines[530], layouts[530], infinityBuf, [buf_384, buf_381], [1950, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[531], layouts[531], infinityBuf, [buf_385, buf_372, buf_381, buf_378, buf_384], [100, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[532], layouts[532], infinityBuf, [buf_386, buf_385, buf_319, buf_311, buf_309, buf_381], [300, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[533], layouts[533], infinityBuf, [output0, buf_386], [300, 1, 1]);
        commandEncoder.copyBufferToBuffer(output0, 0, gpuReadBuffer0, 0, output0.size);
        const gpuCommands = commandEncoder.finish();
        device.queue.submit([gpuCommands]);

        await gpuReadBuffer0.mapAsync(GPUMapMode.READ);
        const resultBuffer0 = new Float32Array(gpuReadBuffer0.size/4);
        resultBuffer0.set(new Float32Array(gpuReadBuffer0.getMappedRange()));
        gpuReadBuffer0.unmap();
        return [resultBuffer0];
    }
}
const load = async (device, weight_path) => { return await fetch(weight_path).then(x => x.arrayBuffer()).then(x => setupNet(device, new Uint8Array(x))); }
return { load, setupNet };
})();
export default RFDETR;
