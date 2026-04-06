
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
@compute @workgroup_size(32,3) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 4608 */
  var lidx0 = i32(lindex.x); /* 32 */
  var lidx1 = i32(lindex.y); /* 3 */
  var alu0 = ((gidx0*96)+(lidx0*3));
  var val0 = data1_442368[((alu0-lidx1)+2)];
  data0_442368[(lidx1+alu0)] = (val0*0.00392156862745098f);
}`;

const r_100_16_16_3_256 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_76800:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_998400:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_196608:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_768:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,3>;
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 100 */
  var lidx0 = i32(lindex.x); /* 16 */
  var cast0 = bitcast<u32>(gidx0);
  var alu0 = (gidx1*768);
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 256; Ridx0++) {
    var alu4 = (alu0+Ridx0);
    var val0 = data1_998400[alu4];
    var val1 = data2_196608[(bitcast<i32>((cast0<<12u))+bitcast<i32>((bitcast<u32>(lidx0)<<8u))+Ridx0+131072)];
    var val2 = data1_998400[(alu4+256)];
    var val3 = data1_998400[(alu4+512)];
    acc0[0] = (acc0[0]+(val0*val1));
    acc0[1] = (acc0[1]+(val2*val1));
    acc0[2] = (acc0[2]+(val3*val1));
  }
  var alu9 = (lidx0+bitcast<i32>((cast0<<4u)));
  var val4 = data3_768[(alu9+512)];
  var alu10 = (alu9+alu0);
  data0_76800[alu10] = (acc0[0]+val4);
  data0_76800[(alu10+256)] = (acc0[1]+val4);
  data0_76800[(alu10+512)] = (acc0[2]+val4);
}`;

const E_9216_16_3 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_442368:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_442368:array<f32>;
@compute @workgroup_size(16,3) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 9216 */
  var lidx0 = i32(lindex.x); /* 16 */
  var lidx1 = i32(lindex.y); /* 3 */
  var alu0 = ((gidx0*48)+(lidx0*3));
  var val0 = data1_442368[(lidx1+alu0)];
  data0_442368[((alu0-lidx1)+2)] = val0;
}`;

const E_384_24_16_3 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_442368:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_442368:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 24 */
  var gidx1 = i32(gindex.y); /* 384 */
  var lidx0 = i32(lindex.x); /* 16 */
  var alu0 = (lidx0+bitcast<i32>((bitcast<u32>(gidx0)<<4u)));
  var alu1 = ((f32((alu0+1)))+-1.0f);
  var alu2 = select(alu1,383.0f,(383.0f<alu1));
  var alu3 = trunc(alu2);
  var cast0 = (i32(alu3));
  var alu4 = (gidx1*1152);
  var alu5 = select(cast0,(i32((alu3+1.0f))),(alu3<alu2));
  var alu6 = (alu4+(alu5*3));
  var alu7 = ((-1<alu5)&(alu5<384));
  var val0 = select(0.0f, data1_442368[(alu6+1)], alu7);
  var val1 = select(0.0f, data1_442368[(alu6+2)], alu7);
  var alu8 = (alu3+-1.0f);
  var alu9 = (alu2<alu3);
  var alu10 = select(cast0,(i32(alu8)),alu9);
  var alu11 = (alu4+(alu10*3));
  var alu12 = ((-1<alu10)&(alu10<384));
  var val2 = select(0.0f, data1_442368[(alu11+1)], alu12);
  var val3 = select(0.0f, data1_442368[(alu11+2)], alu12);
  var val4 = select(0.0f, data1_442368[alu6], alu7);
  var val5 = select(0.0f, data1_442368[alu11], alu12);
  var alu13 = (alu0+(gidx1*384));
  var alu14 = select(alu3,alu8,alu9);
  var alu15 = (alu2-alu14);
  data0_442368[alu13] = (val3+((val1-val3)*alu15));
  data0_442368[(alu13+147456)] = (val2+((val0-val2)*alu15));
  data0_442368[(alu13+294912)] = (val5+((val4-val5)*alu15));
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

const r_577_24_8_8_2_16_2_3 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32,128>;
@group(0) @binding(1)var<storage,read_write>data0_221568:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_384:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_442368:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_294912:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_384:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_221568:array<f32>;
@compute @workgroup_size(8,8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,2>;
  var acc1: array<f32,2>;
  var gidx0 = i32(gindex.x); /* 24 */
  var gidx1 = i32(gindex.y); /* 577 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 8 */
  var alu0 = (gidx1+23);
  var alu1 = ((alu0*683)>>14u);
  var alu2 = (0<gidx1);
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  for (var Ridx1 = 0; Ridx1 < 16; Ridx1++) {
    for (var Ridx2 = 0; Ridx2 < 2; Ridx2++) {
      var alu5 = ((lidx1*3)+(Ridx2*24)+((alu0-(24*alu1))*48)+(alu1*18432)+(Ridx1*1152));
      var val0 = select(0.0f, data2_442368[(alu5+-18432)], alu2);
      var alu6 = (lidx1+bitcast<i32>((bitcast<u32>(Ridx2)<<3u))+bitcast<i32>((bitcast<u32>(Ridx1)<<4u))+(gidx0*12288)+(lidx0*1536));
      var val1 = data3_294912[alu6];
      var val2 = select(0.0f, data2_442368[(alu5+-18431)], alu2);
      var val3 = data3_294912[(alu6+256)];
      var val4 = select(0.0f, data2_442368[(alu5+-18430)], alu2);
      var val5 = data3_294912[(alu6+512)];
      var val6 = data3_294912[(alu6+768)];
      var val7 = data3_294912[(alu6+1024)];
      var val8 = data3_294912[(alu6+1280)];
      acc0[0] = (acc0[0]+(val0*val1)+(val2*val3)+(val4*val5));
      acc0[1] = (acc0[1]+(val0*val6)+(val2*val7)+(val4*val8));
    }
  }
  var cast0 = bitcast<u32>(lidx0);
  var cast1 = bitcast<i32>((cast0<<4u));
  var alu11 = (cast1+bitcast<i32>((bitcast<u32>(lidx1)<<1u)));
  temp0[alu11] = acc0[0];
  temp0[(alu11+1)] = acc0[1];
  workgroupBarrier();
  acc1[0] = 0.0f;
  acc1[1] = 0.0f;
  for (var Ridx106 = 0; Ridx106 < 8; Ridx106++) {
    var alu17 = (cast1+bitcast<i32>((bitcast<u32>(Ridx106)<<1u)));
    var val9 = temp0[alu17];
    var val10 = temp0[(alu17+1)];
    acc1[0] = (acc1[0]+val9);
    acc1[1] = (acc1[1]+val10);
  }
  var alu21 = (bitcast<i32>((bitcast<u32>(gidx0)<<4u))+bitcast<i32>((cast0<<1u)));
  var val11 = data1_384[alu21];
  var alu22 = (alu21+1);
  var val12 = data1_384[alu22];
  var val13 = data4_384[alu21];
  var alu23 = (alu21+(gidx1*384));
  var val14 = data5_221568[alu23];
  var val15 = data4_384[alu22];
  var alu24 = (alu23+1);
  var val16 = data5_221568[alu24];
  var alu25 = (lidx1==0);
  var alu26 = (gidx1<1);
  var alu27 = select(0.0f,val11,alu26);
  var alu28 = select((acc1[0]+val13),0.0f,alu26);
  var alu29 = select(0.0f,val12,alu26);
  var alu30 = select((acc1[1]+val15),0.0f,alu26);
  if (alu25) {
    data0_221568[alu23] = (alu27+alu28+val14);
  }
  if (alu25) {
    data0_221568[alu24] = (alu29+alu30+val16);
  }
}`;

const r_2_2_145_16_24 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32,16>;
@group(0) @binding(1)var<storage,read_write>data0_580:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_221568:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,1>;
  var acc1: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 145 */
  var gidx1 = i32(gindex.y); /* 2 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 16 */
  var alu0 = (gidx0+11);
  var alu1 = ((alu0*171)>>11u);
  acc0[0] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 24; Ridx0++) {
    var alu3 = ((lidx0*24)+Ridx0);
    var val0 = select(0.0f, data1_221568[((gidx1*4608)+((alu0-(12*alu1))*384)+(alu1*9216)+(gidx2*110592)+alu3+-8832)], (0<gidx0));
    var val1 = data1_221568[alu3];
    var alu4 = select(0.0f,val1,(gidx0<1));
    acc0[0] = (acc0[0]+alu4+val0);
  }
  temp0[lidx0] = acc0[0];
  workgroupBarrier();
  acc1[0] = 0.0f;
  for (var Ridx103 = 0; Ridx103 < 16; Ridx103++) {
    var val2 = temp0[Ridx103];
    acc1[0] = (acc1[0]+val2);
  }
  var alu12 = (lidx0==0);
  if (alu12) {
    data0_580[(gidx0+(gidx1*145)+(gidx2*290))] = (acc1[0]*0.0026041666666666665f);
  }
}`;

const r_2_2_145_32_12 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
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

const r_116_24_16_5_384 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_222720:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_222720:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_147456:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_384:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,5>;
  var gidx0 = i32(gindex.x); /* 24 */
  var gidx1 = i32(gindex.y); /* 116 */
  var lidx0 = i32(lindex.x); /* 16 */
  var alu0 = (gidx1*1920);
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  acc0[4] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 384; Ridx0++) {
    var alu6 = (alu0+Ridx0);
    var val0 = data1_222720[alu6];
    var val1 = data2_147456[((gidx0*6144)+(lidx0*384)+Ridx0)];
    var val2 = data1_222720[(alu6+384)];
    var val3 = data1_222720[(alu6+768)];
    var val4 = data1_222720[(alu6+1152)];
    var val5 = data1_222720[(alu6+1536)];
    acc0[0] = (acc0[0]+(val0*val1));
    acc0[1] = (acc0[1]+(val2*val1));
    acc0[2] = (acc0[2]+(val3*val1));
    acc0[3] = (acc0[3]+(val4*val1));
    acc0[4] = (acc0[4]+(val5*val1));
  }
  var alu13 = (lidx0+bitcast<i32>((bitcast<u32>(gidx0)<<4u)));
  var val6 = data3_384[alu13];
  var alu14 = (alu13+alu0);
  data0_222720[alu14] = (acc0[0]+val6);
  data0_222720[(alu14+384)] = (acc0[1]+val6);
  data0_222720[(alu14+768)] = (acc0[2]+val6);
  data0_222720[(alu14+1152)] = (acc0[3]+val6);
  data0_222720[(alu14+1536)] = (acc0[4]+val6);
}`;

const r_4_6_5_145_29_64 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_504600:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_222720:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_222720:array<f32>;
@compute @workgroup_size(29) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 725 */
  var gidx1 = i32(gindex.y); /* 6 */
  var gidx2 = i32(gindex.z); /* 4 */
  var lidx0 = i32(lindex.x); /* 29 */
  var alu0 = ((gidx0*113)>>14u);
  var alu1 = (gidx2*55680);
  var alu2 = (gidx0-(145*alu0));
  acc0[0] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 64; Ridx0++) {
    var alu4 = (bitcast<i32>((bitcast<u32>(gidx1)<<6u))+Ridx0);
    var val0 = data1_222720[(alu4+(lidx0*384)+(alu0*11136)+alu1)];
    var val1 = data2_222720[(alu4+(alu2*384)+alu1)];
    acc0[0] = (acc0[0]+(val0*val1));
  }
  data0_504600[((lidx0*145)+(alu0*4205)+alu2+(gidx1*21025)+(gidx2*126150))] = (acc0[0]*0.125f);
}`;

const r_435_8_145 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_3480:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_504600:array<f32>;
@compute @workgroup_size(8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 435 */
  var lidx0 = i32(lindex.x); /* 8 */
  acc0[0] = (f32(-INFINITY));
  for (var Ridx0 = 0; Ridx0 < 145; Ridx0++) {
    var val0 = data1_504600[((gidx0*1160)+(lidx0*145)+Ridx0)];
    var alu1 = select(acc0[0],val0,(acc0[0]<val0));
    acc0[0] = alu1;
  }
  data0_3480[(lidx0+bitcast<i32>((bitcast<u32>(gidx0)<<3u)))] = acc0[0];
}`;

const r_435_8_145n1 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_3480:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_504600:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_3480:array<f32>;
@compute @workgroup_size(8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 435 */
  var lidx0 = i32(lindex.x); /* 8 */
  var alu0 = (lidx0+bitcast<i32>((bitcast<u32>(gidx0)<<3u)));
  var val0 = data2_3480[alu0];
  acc0[0] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 145; Ridx0++) {
    var val1 = data1_504600[((gidx0*1160)+(lidx0*145)+Ridx0)];
    acc0[0] = (acc0[0]+exp2(((val1-val0)*1.4426950408889634f)));
  }
  data0_3480[alu0] = acc0[0];
}`;

const r_4_145_16_2_3_4_145 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_222720:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_504600:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_3480:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_3480:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_222720:array<f32>;
@compute @workgroup_size(16,2,3) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 145 */
  var gidx1 = i32(gindex.y); /* 4 */
  var lidx1 = i32(lindex.y); /* 2 */
  var lidx2 = i32(lindex.z); /* 3 */
  var alu0 = (gidx0+(lidx1*145)+(lidx2*290)+(gidx1*870));
  var val0 = data2_3480[alu0];
  var lidx0 = i32(lindex.x); /* 16 */
  var alu1 = (lidx0+bitcast<i32>((bitcast<u32>(lidx1)<<6u))+bitcast<i32>((bitcast<u32>(lidx2)<<7u)));
  var alu2 = (gidx1*55680);
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 145; Ridx0++) {
    var val1 = data1_504600[((gidx0*145)+Ridx0+(lidx1*21025)+(lidx2*42050)+(gidx1*126150))];
    var alu7 = (alu1+(Ridx0*384)+alu2);
    var val2 = data4_222720[alu7];
    var val3 = data4_222720[(alu7+16)];
    var val4 = data4_222720[(alu7+32)];
    var val5 = data4_222720[(alu7+48)];
    var alu8 = exp2(((val1-val0)*1.4426950408889634f));
    acc0[0] = (acc0[0]+(alu8*val2));
    acc0[1] = (acc0[1]+(alu8*val3));
    acc0[2] = (acc0[2]+(alu8*val4));
    acc0[3] = (acc0[3]+(alu8*val5));
  }
  var val6 = data3_3480[alu0];
  var alu14 = (alu1+(gidx0*384)+alu2);
  var alu15 = (1/val6);
  data0_222720[alu14] = (acc0[0]*alu15);
  data0_222720[(alu14+16)] = (acc0[1]*alu15);
  data0_222720[(alu14+32)] = (acc0[2]*alu15);
  data0_222720[(alu14+48)] = (acc0[3]*alu15);
}`;

const r_2_2_29_12_16_4_5_2_96 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32,640>;
@group(0) @binding(1)var<storage,read_write>data0_222720:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_222720:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_147456:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_384:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_384:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_221568:array<f32>;
@compute @workgroup_size(16,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,10>;
  var acc1: array<f32,10>;
  var gidx0 = i32(gindex.x); /* 348 */
  var gidx1 = i32(gindex.y); /* 2 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 16 */
  var lidx1 = i32(lindex.y); /* 4 */
  var alu0 = ((gidx1*55680)+(gidx2*111360));
  var alu1 = ((gidx0*171)>>11u);
  var alu2 = (gidx0-(12*alu1));
  var alu3 = (alu1*1920);
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
  for (var Ridx0 = 0; Ridx0 < 96; Ridx0++) {
    var alu14 = (lidx1+bitcast<i32>((bitcast<u32>(Ridx0)<<2u)));
    var alu15 = (alu14+alu3+alu0);
    var val0 = data1_222720[alu15];
    var alu16 = (alu14+(lidx0*384)+(alu2*12288));
    var val1 = data2_147456[alu16];
    var val2 = data2_147456[(alu16+6144)];
    var val3 = data1_222720[(alu15+384)];
    var val4 = data1_222720[(alu15+768)];
    var val5 = data1_222720[(alu15+1152)];
    var val6 = data1_222720[(alu15+1536)];
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
  var alu28 = (lidx0*40);
  var alu29 = (alu28+(lidx1*10));
  temp0[(alu29+1)] = acc0[1];
  temp0[(alu29+2)] = acc0[2];
  temp0[(alu29+3)] = acc0[3];
  temp0[(alu29+4)] = acc0[4];
  temp0[(alu29+5)] = acc0[5];
  temp0[(alu29+6)] = acc0[6];
  temp0[(alu29+7)] = acc0[7];
  temp0[(alu29+8)] = acc0[8];
  temp0[(alu29+9)] = acc0[9];
  temp0[alu29] = acc0[0];
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
  for (var Ridx107 = 0; Ridx107 < 4; Ridx107++) {
    var alu51 = (alu28+(Ridx107*10));
    var val7 = temp0[(alu51+1)];
    var val8 = temp0[(alu51+2)];
    var val9 = temp0[alu51];
    var val10 = temp0[(alu51+3)];
    var val11 = temp0[(alu51+4)];
    var val12 = temp0[(alu51+5)];
    var val13 = temp0[(alu51+6)];
    var val14 = temp0[(alu51+7)];
    var val15 = temp0[(alu51+8)];
    var val16 = temp0[(alu51+9)];
    acc1[0] = (acc1[0]+val9);
    acc1[1] = (acc1[1]+val7);
    acc1[2] = (acc1[2]+val8);
    acc1[3] = (acc1[3]+val10);
    acc1[4] = (acc1[4]+val11);
    acc1[5] = (acc1[5]+val12);
    acc1[6] = (acc1[6]+val13);
    acc1[7] = (acc1[7]+val14);
    acc1[8] = (acc1[8]+val15);
    acc1[9] = (acc1[9]+val16);
  }
  var alu63 = (lidx0+bitcast<i32>((bitcast<u32>(alu2)<<5u)));
  var val17 = data3_384[alu63];
  var val18 = data4_384[alu63];
  var val19 = data5_221568[alu63];
  var alu64 = (alu1*5);
  var alu65 = (alu64+11);
  var alu66 = (gidx1*4608);
  var alu67 = ((alu65*171)>>11u);
  var alu68 = (gidx2*110592);
  var alu69 = (alu63+alu66+((alu65-(12*alu67))*384)+(alu67*9216)+alu68);
  var alu70 = (11<gidx0);
  var val20 = select(0.0f, data5_221568[(alu69+-8832)], alu70);
  var alu71 = (alu63+16);
  var val21 = data3_384[alu71];
  var val22 = data4_384[alu71];
  var val23 = data5_221568[alu71];
  var val24 = select(0.0f, data5_221568[(alu69+-8816)], alu70);
  var alu72 = ((alu64*171)>>11u);
  var alu73 = (alu63+alu66+((alu64-(12*alu72))*384)+(alu72*9216)+alu68);
  var val25 = data5_221568[(alu73+384)];
  var alu74 = (alu64+1);
  var alu75 = ((alu74*171)>>11u);
  var alu76 = (alu63+alu66+((alu74-(12*alu75))*384)+(alu75*9216)+alu68);
  var val26 = data5_221568[(alu76+384)];
  var val27 = data5_221568[(alu76+400)];
  var val28 = data5_221568[(alu73+400)];
  var alu77 = (alu64+2);
  var alu78 = ((alu77*171)>>11u);
  var alu79 = (alu63+alu66+((alu77-(12*alu78))*384)+(alu78*9216)+alu68);
  var val29 = data5_221568[(alu79+384)];
  var val30 = data5_221568[(alu79+400)];
  var alu80 = (alu64+3);
  var alu81 = ((alu80*171)>>11u);
  var alu82 = (alu63+alu66+((alu80-(12*alu81))*384)+(alu81*9216)+alu68);
  var val31 = data5_221568[(alu82+384)];
  var val32 = data5_221568[(alu82+400)];
  var alu83 = (alu63+alu3+alu0);
  var alu84 = (lidx1==0);
  var alu85 = (gidx0<12);
  var alu86 = select(0.0f,val19,alu85);
  var alu87 = select(0.0f,val23,alu85);
  if (alu84) {
    data0_222720[alu83] = (((acc1[0]+val17)*val18)+alu86+val20);
  }
  if (alu84) {
    data0_222720[(alu83+16)] = (((acc1[1]+val21)*val22)+alu87+val24);
  }
  if (alu84) {
    data0_222720[(alu83+384)] = (((acc1[2]+val17)*val18)+val25);
  }
  if (alu84) {
    data0_222720[(alu83+400)] = (((acc1[3]+val21)*val22)+val28);
  }
  if (alu84) {
    data0_222720[(alu83+768)] = (((acc1[4]+val17)*val18)+val26);
  }
  if (alu84) {
    data0_222720[(alu83+784)] = (((acc1[5]+val21)*val22)+val27);
  }
  if (alu84) {
    data0_222720[(alu83+1152)] = (((acc1[6]+val17)*val18)+val29);
  }
  if (alu84) {
    data0_222720[(alu83+1168)] = (((acc1[7]+val21)*val22)+val30);
  }
  if (alu84) {
    data0_222720[(alu83+1536)] = (((acc1[8]+val17)*val18)+val31);
  }
  if (alu84) {
    data0_222720[(alu83+1552)] = (((acc1[9]+val21)*val22)+val32);
  }
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

const r_20_32_29_4_4_3_384 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_890880:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_222720:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_589824:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_1536:array<f32>;
@compute @workgroup_size(29) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,48>;
  var gidx0 = i32(gindex.x); /* 32 */
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
    var val0 = data1_222720[((gidx1*11136)+(lidx0*384)+Ridx0)];
    var alu48 = ((gidx0*18432)+Ridx0);
    var val1 = data2_589824[(alu48+6144)];
    var val2 = data2_589824[alu48];
    var val3 = data2_589824[(alu48+12288)];
    var val4 = data2_589824[(alu48+1536)];
    var val5 = data2_589824[(alu48+7680)];
    var val6 = data2_589824[(alu48+13824)];
    var val7 = data2_589824[(alu48+3072)];
    var val8 = data2_589824[(alu48+9216)];
    var val9 = data2_589824[(alu48+15360)];
    var val10 = data2_589824[(alu48+10752)];
    var val11 = data2_589824[(alu48+16896)];
    var val12 = data2_589824[(alu48+384)];
    var val13 = data2_589824[(alu48+4608)];
    var val14 = data2_589824[(alu48+6528)];
    var val15 = data2_589824[(alu48+1920)];
    var val16 = data2_589824[(alu48+8064)];
    var val17 = data2_589824[(alu48+12672)];
    var val18 = data2_589824[(alu48+3456)];
    var val19 = data2_589824[(alu48+4992)];
    var val20 = data2_589824[(alu48+9600)];
    var val21 = data2_589824[(alu48+14208)];
    var val22 = data2_589824[(alu48+15744)];
    var val23 = data2_589824[(alu48+11136)];
    var val24 = data2_589824[(alu48+17280)];
    var val25 = data2_589824[(alu48+768)];
    var val26 = data2_589824[(alu48+6912)];
    var val27 = data2_589824[(alu48+13056)];
    var val28 = data2_589824[(alu48+2304)];
    var val29 = data2_589824[(alu48+8448)];
    var val30 = data2_589824[(alu48+14592)];
    var val31 = data2_589824[(alu48+3840)];
    var val32 = data2_589824[(alu48+9984)];
    var val33 = data2_589824[(alu48+16128)];
    var val34 = data2_589824[(alu48+5376)];
    var val35 = data2_589824[(alu48+17664)];
    var val36 = data2_589824[(alu48+1152)];
    var val37 = data2_589824[(alu48+7296)];
    var val38 = data2_589824[(alu48+11520)];
    var val39 = data2_589824[(alu48+2688)];
    var val40 = data2_589824[(alu48+8832)];
    var val41 = data2_589824[(alu48+14976)];
    var val42 = data2_589824[(alu48+4224)];
    var val43 = data2_589824[(alu48+10368)];
    var val44 = data2_589824[(alu48+5760)];
    var val45 = data2_589824[(alu48+11904)];
    var val46 = data2_589824[(alu48+13440)];
    var val47 = data2_589824[(alu48+16512)];
    var val48 = data2_589824[(alu48+18048)];
    acc0[0] = (acc0[0]+(val0*val2));
    acc0[1] = (acc0[1]+(val0*val1));
    acc0[2] = (acc0[2]+(val0*val3));
    acc0[3] = (acc0[3]+(val0*val4));
    acc0[4] = (acc0[4]+(val0*val5));
    acc0[5] = (acc0[5]+(val0*val6));
    acc0[6] = (acc0[6]+(val0*val7));
    acc0[7] = (acc0[7]+(val0*val8));
    acc0[8] = (acc0[8]+(val0*val9));
    acc0[9] = (acc0[9]+(val0*val13));
    acc0[10] = (acc0[10]+(val0*val10));
    acc0[11] = (acc0[11]+(val0*val11));
    acc0[12] = (acc0[12]+(val0*val12));
    acc0[13] = (acc0[13]+(val0*val14));
    acc0[14] = (acc0[14]+(val0*val17));
    acc0[15] = (acc0[15]+(val0*val15));
    acc0[16] = (acc0[16]+(val0*val16));
    acc0[17] = (acc0[17]+(val0*val21));
    acc0[18] = (acc0[18]+(val0*val18));
    acc0[19] = (acc0[19]+(val0*val20));
    acc0[20] = (acc0[20]+(val0*val22));
    acc0[21] = (acc0[21]+(val0*val19));
    acc0[22] = (acc0[22]+(val0*val23));
    acc0[23] = (acc0[23]+(val0*val24));
    acc0[24] = (acc0[24]+(val0*val25));
    acc0[25] = (acc0[25]+(val0*val26));
    acc0[26] = (acc0[26]+(val0*val27));
    acc0[27] = (acc0[27]+(val0*val28));
    acc0[28] = (acc0[28]+(val0*val29));
    acc0[29] = (acc0[29]+(val0*val30));
    acc0[30] = (acc0[30]+(val0*val31));
    acc0[31] = (acc0[31]+(val0*val32));
    acc0[32] = (acc0[32]+(val0*val33));
    acc0[33] = (acc0[33]+(val0*val34));
    acc0[34] = (acc0[34]+(val0*val38));
    acc0[35] = (acc0[35]+(val0*val35));
    acc0[36] = (acc0[36]+(val0*val36));
    acc0[37] = (acc0[37]+(val0*val37));
    acc0[38] = (acc0[38]+(val0*val46));
    acc0[39] = (acc0[39]+(val0*val39));
    acc0[40] = (acc0[40]+(val0*val40));
    acc0[41] = (acc0[41]+(val0*val41));
    acc0[42] = (acc0[42]+(val0*val42));
    acc0[43] = (acc0[43]+(val0*val43));
    acc0[44] = (acc0[44]+(val0*val47));
    acc0[45] = (acc0[45]+(val0*val44));
    acc0[46] = (acc0[46]+(val0*val45));
    acc0[47] = (acc0[47]+(val0*val48));
  }
  var alu98 = (gidx0*48);
  var val49 = data3_1536[alu98];
  var val50 = data3_1536[(alu98+1)];
  var val51 = data3_1536[(alu98+2)];
  var val52 = data3_1536[(alu98+3)];
  var val53 = data3_1536[(alu98+4)];
  var val54 = data3_1536[(alu98+5)];
  var val55 = data3_1536[(alu98+6)];
  var val56 = data3_1536[(alu98+7)];
  var val57 = data3_1536[(alu98+8)];
  var val58 = data3_1536[(alu98+9)];
  var val59 = data3_1536[(alu98+10)];
  var val60 = data3_1536[(alu98+11)];
  var val61 = data3_1536[(alu98+12)];
  var val62 = data3_1536[(alu98+13)];
  var val63 = data3_1536[(alu98+14)];
  var val64 = data3_1536[(alu98+15)];
  var val65 = data3_1536[(alu98+16)];
  var val66 = data3_1536[(alu98+17)];
  var val67 = data3_1536[(alu98+18)];
  var val68 = data3_1536[(alu98+19)];
  var val69 = data3_1536[(alu98+20)];
  var val70 = data3_1536[(alu98+21)];
  var val71 = data3_1536[(alu98+22)];
  var val72 = data3_1536[(alu98+23)];
  var val73 = data3_1536[(alu98+24)];
  var val74 = data3_1536[(alu98+25)];
  var val75 = data3_1536[(alu98+26)];
  var val76 = data3_1536[(alu98+27)];
  var val77 = data3_1536[(alu98+28)];
  var val78 = data3_1536[(alu98+29)];
  var val79 = data3_1536[(alu98+30)];
  var val80 = data3_1536[(alu98+31)];
  var val81 = data3_1536[(alu98+32)];
  var val82 = data3_1536[(alu98+33)];
  var val83 = data3_1536[(alu98+34)];
  var val84 = data3_1536[(alu98+35)];
  var val85 = data3_1536[(alu98+36)];
  var val86 = data3_1536[(alu98+37)];
  var val87 = data3_1536[(alu98+38)];
  var val88 = data3_1536[(alu98+39)];
  var val89 = data3_1536[(alu98+40)];
  var val90 = data3_1536[(alu98+41)];
  var val91 = data3_1536[(alu98+42)];
  var val92 = data3_1536[(alu98+43)];
  var val93 = data3_1536[(alu98+44)];
  var val94 = data3_1536[(alu98+45)];
  var val95 = data3_1536[(alu98+46)];
  var val96 = data3_1536[(alu98+47)];
  var alu99 = ((gidx1*44544)+(lidx0*1536)+alu98);
  var alu100 = (acc0[0]+val49);
  var alu101 = (acc0[1]+val65);
  var alu102 = (acc0[2]+val81);
  var alu103 = (acc0[3]+val53);
  var alu104 = (acc0[4]+val69);
  var alu105 = (acc0[5]+val85);
  var alu106 = (acc0[6]+val57);
  var alu107 = (acc0[7]+val73);
  var alu108 = (acc0[8]+val89);
  var alu109 = (acc0[9]+val61);
  var alu110 = (acc0[10]+val77);
  var alu111 = (acc0[11]+val93);
  var alu112 = (acc0[12]+val50);
  var alu113 = (acc0[13]+val66);
  var alu114 = (acc0[14]+val82);
  var alu115 = (acc0[15]+val54);
  var alu116 = (acc0[16]+val70);
  var alu117 = (acc0[17]+val86);
  var alu118 = (acc0[18]+val58);
  var alu119 = (acc0[19]+val74);
  var alu120 = (acc0[20]+val90);
  var alu121 = (acc0[21]+val62);
  var alu122 = (acc0[22]+val78);
  var alu123 = (acc0[23]+val94);
  var alu124 = (acc0[24]+val51);
  var alu125 = (acc0[25]+val67);
  var alu126 = (acc0[26]+val83);
  var alu127 = (acc0[27]+val55);
  var alu128 = (acc0[28]+val71);
  var alu129 = (acc0[29]+val87);
  var alu130 = (acc0[30]+val59);
  var alu131 = (acc0[31]+val75);
  var alu132 = (acc0[32]+val91);
  var alu133 = (acc0[33]+val63);
  var alu134 = (acc0[34]+val79);
  var alu135 = (acc0[35]+val95);
  var alu136 = (acc0[36]+val52);
  var alu137 = (acc0[37]+val68);
  var alu138 = (acc0[38]+val84);
  var alu139 = (acc0[39]+val56);
  var alu140 = (acc0[40]+val72);
  var alu141 = (acc0[41]+val88);
  var alu142 = (acc0[42]+val60);
  var alu143 = (acc0[43]+val76);
  var alu144 = (acc0[44]+val92);
  var alu145 = (acc0[45]+val64);
  var alu146 = (acc0[46]+val80);
  var alu147 = (acc0[47]+val96);
  var alu148 = (alu100*0.7071067811865475f);
  var alu149 = select(1.0f,-1.0f,(alu148<0.0f));
  var alu150 = select(0.0f,alu149,(alu148!=0.0f));
  var alu151 = (1/(1.0f+(alu100*alu150*0.2316418882663604f)));
  var alu152 = (alu101*0.7071067811865475f);
  var alu153 = select(1.0f,-1.0f,(alu152<0.0f));
  var alu154 = select(0.0f,alu153,(alu152!=0.0f));
  var alu155 = (1/(1.0f+(alu101*alu154*0.2316418882663604f)));
  var alu156 = (alu102*0.7071067811865475f);
  var alu157 = select(1.0f,-1.0f,(alu156<0.0f));
  var alu158 = select(0.0f,alu157,(alu156!=0.0f));
  var alu159 = (1/(1.0f+(alu102*alu158*0.2316418882663604f)));
  var alu160 = (alu103*0.7071067811865475f);
  var alu161 = select(1.0f,-1.0f,(alu160<0.0f));
  var alu162 = select(0.0f,alu161,(alu160!=0.0f));
  var alu163 = (1/(1.0f+(alu103*alu162*0.2316418882663604f)));
  var alu164 = (alu104*0.7071067811865475f);
  var alu165 = select(1.0f,-1.0f,(alu164<0.0f));
  var alu166 = select(0.0f,alu165,(alu164!=0.0f));
  var alu167 = (1/(1.0f+(alu104*alu166*0.2316418882663604f)));
  var alu168 = (alu105*0.7071067811865475f);
  var alu169 = select(1.0f,-1.0f,(alu168<0.0f));
  var alu170 = select(0.0f,alu169,(alu168!=0.0f));
  var alu171 = (1/(1.0f+(alu105*alu170*0.2316418882663604f)));
  var alu172 = (alu106*0.7071067811865475f);
  var alu173 = select(1.0f,-1.0f,(alu172<0.0f));
  var alu174 = select(0.0f,alu173,(alu172!=0.0f));
  var alu175 = (1/(1.0f+(alu106*alu174*0.2316418882663604f)));
  var alu176 = (alu107*0.7071067811865475f);
  var alu177 = select(1.0f,-1.0f,(alu176<0.0f));
  var alu178 = select(0.0f,alu177,(alu176!=0.0f));
  var alu179 = (1/(1.0f+(alu107*alu178*0.2316418882663604f)));
  var alu180 = (alu108*0.7071067811865475f);
  var alu181 = select(1.0f,-1.0f,(alu180<0.0f));
  var alu182 = select(0.0f,alu181,(alu180!=0.0f));
  var alu183 = (1/(1.0f+(alu108*alu182*0.2316418882663604f)));
  var alu184 = (alu109*0.7071067811865475f);
  var alu185 = select(1.0f,-1.0f,(alu184<0.0f));
  var alu186 = select(0.0f,alu185,(alu184!=0.0f));
  var alu187 = (1/(1.0f+(alu109*alu186*0.2316418882663604f)));
  var alu188 = (alu110*0.7071067811865475f);
  var alu189 = select(1.0f,-1.0f,(alu188<0.0f));
  var alu190 = select(0.0f,alu189,(alu188!=0.0f));
  var alu191 = (1/(1.0f+(alu110*alu190*0.2316418882663604f)));
  var alu192 = (alu111*0.7071067811865475f);
  var alu193 = select(1.0f,-1.0f,(alu192<0.0f));
  var alu194 = select(0.0f,alu193,(alu192!=0.0f));
  var alu195 = (1/(1.0f+(alu111*alu194*0.2316418882663604f)));
  var alu196 = (alu112*0.7071067811865475f);
  var alu197 = select(1.0f,-1.0f,(alu196<0.0f));
  var alu198 = select(0.0f,alu197,(alu196!=0.0f));
  var alu199 = (1/(1.0f+(alu112*alu198*0.2316418882663604f)));
  var alu200 = (alu113*0.7071067811865475f);
  var alu201 = select(1.0f,-1.0f,(alu200<0.0f));
  var alu202 = select(0.0f,alu201,(alu200!=0.0f));
  var alu203 = (1/(1.0f+(alu113*alu202*0.2316418882663604f)));
  var alu204 = (alu114*0.7071067811865475f);
  var alu205 = select(1.0f,-1.0f,(alu204<0.0f));
  var alu206 = select(0.0f,alu205,(alu204!=0.0f));
  var alu207 = (1/(1.0f+(alu114*alu206*0.2316418882663604f)));
  var alu208 = (alu115*0.7071067811865475f);
  var alu209 = select(1.0f,-1.0f,(alu208<0.0f));
  var alu210 = select(0.0f,alu209,(alu208!=0.0f));
  var alu211 = (1/(1.0f+(alu115*alu210*0.2316418882663604f)));
  var alu212 = (alu116*0.7071067811865475f);
  var alu213 = select(1.0f,-1.0f,(alu212<0.0f));
  var alu214 = select(0.0f,alu213,(alu212!=0.0f));
  var alu215 = (1/(1.0f+(alu116*alu214*0.2316418882663604f)));
  var alu216 = (alu117*0.7071067811865475f);
  var alu217 = select(1.0f,-1.0f,(alu216<0.0f));
  var alu218 = select(0.0f,alu217,(alu216!=0.0f));
  var alu219 = (1/(1.0f+(alu117*alu218*0.2316418882663604f)));
  var alu220 = (alu118*0.7071067811865475f);
  var alu221 = select(1.0f,-1.0f,(alu220<0.0f));
  var alu222 = select(0.0f,alu221,(alu220!=0.0f));
  var alu223 = (1/(1.0f+(alu118*alu222*0.2316418882663604f)));
  var alu224 = (alu119*0.7071067811865475f);
  var alu225 = select(1.0f,-1.0f,(alu224<0.0f));
  var alu226 = select(0.0f,alu225,(alu224!=0.0f));
  var alu227 = (1/(1.0f+(alu119*alu226*0.2316418882663604f)));
  var alu228 = (alu120*0.7071067811865475f);
  var alu229 = select(1.0f,-1.0f,(alu228<0.0f));
  var alu230 = select(0.0f,alu229,(alu228!=0.0f));
  var alu231 = (1/(1.0f+(alu120*alu230*0.2316418882663604f)));
  var alu232 = (alu121*0.7071067811865475f);
  var alu233 = select(1.0f,-1.0f,(alu232<0.0f));
  var alu234 = select(0.0f,alu233,(alu232!=0.0f));
  var alu235 = (1/(1.0f+(alu121*alu234*0.2316418882663604f)));
  var alu236 = (alu122*0.7071067811865475f);
  var alu237 = select(1.0f,-1.0f,(alu236<0.0f));
  var alu238 = select(0.0f,alu237,(alu236!=0.0f));
  var alu239 = (1/(1.0f+(alu122*alu238*0.2316418882663604f)));
  var alu240 = (alu123*0.7071067811865475f);
  var alu241 = select(1.0f,-1.0f,(alu240<0.0f));
  var alu242 = select(0.0f,alu241,(alu240!=0.0f));
  var alu243 = (1/(1.0f+(alu123*alu242*0.2316418882663604f)));
  var alu244 = (alu124*0.7071067811865475f);
  var alu245 = select(1.0f,-1.0f,(alu244<0.0f));
  var alu246 = select(0.0f,alu245,(alu244!=0.0f));
  var alu247 = (1/(1.0f+(alu124*alu246*0.2316418882663604f)));
  var alu248 = (alu125*0.7071067811865475f);
  var alu249 = select(1.0f,-1.0f,(alu248<0.0f));
  var alu250 = select(0.0f,alu249,(alu248!=0.0f));
  var alu251 = (1/(1.0f+(alu125*alu250*0.2316418882663604f)));
  var alu252 = (alu126*0.7071067811865475f);
  var alu253 = select(1.0f,-1.0f,(alu252<0.0f));
  var alu254 = select(0.0f,alu253,(alu252!=0.0f));
  var alu255 = (1/(1.0f+(alu126*alu254*0.2316418882663604f)));
  var alu256 = (alu127*0.7071067811865475f);
  var alu257 = select(1.0f,-1.0f,(alu256<0.0f));
  var alu258 = select(0.0f,alu257,(alu256!=0.0f));
  var alu259 = (1/(1.0f+(alu127*alu258*0.2316418882663604f)));
  var alu260 = (alu128*0.7071067811865475f);
  var alu261 = select(1.0f,-1.0f,(alu260<0.0f));
  var alu262 = select(0.0f,alu261,(alu260!=0.0f));
  var alu263 = (1/(1.0f+(alu128*alu262*0.2316418882663604f)));
  var alu264 = (alu129*0.7071067811865475f);
  var alu265 = select(1.0f,-1.0f,(alu264<0.0f));
  var alu266 = select(0.0f,alu265,(alu264!=0.0f));
  var alu267 = (1/(1.0f+(alu129*alu266*0.2316418882663604f)));
  var alu268 = (alu130*0.7071067811865475f);
  var alu269 = select(1.0f,-1.0f,(alu268<0.0f));
  var alu270 = select(0.0f,alu269,(alu268!=0.0f));
  var alu271 = (1/(1.0f+(alu130*alu270*0.2316418882663604f)));
  var alu272 = (alu131*0.7071067811865475f);
  var alu273 = select(1.0f,-1.0f,(alu272<0.0f));
  var alu274 = select(0.0f,alu273,(alu272!=0.0f));
  var alu275 = (1/(1.0f+(alu131*alu274*0.2316418882663604f)));
  var alu276 = (alu132*0.7071067811865475f);
  var alu277 = select(1.0f,-1.0f,(alu276<0.0f));
  var alu278 = select(0.0f,alu277,(alu276!=0.0f));
  var alu279 = (1/(1.0f+(alu132*alu278*0.2316418882663604f)));
  var alu280 = (alu133*0.7071067811865475f);
  var alu281 = select(1.0f,-1.0f,(alu280<0.0f));
  var alu282 = select(0.0f,alu281,(alu280!=0.0f));
  var alu283 = (1/(1.0f+(alu133*alu282*0.2316418882663604f)));
  var alu284 = (alu134*0.7071067811865475f);
  var alu285 = select(1.0f,-1.0f,(alu284<0.0f));
  var alu286 = select(0.0f,alu285,(alu284!=0.0f));
  var alu287 = (1/(1.0f+(alu134*alu286*0.2316418882663604f)));
  var alu288 = (alu135*0.7071067811865475f);
  var alu289 = select(1.0f,-1.0f,(alu288<0.0f));
  var alu290 = select(0.0f,alu289,(alu288!=0.0f));
  var alu291 = (1/(1.0f+(alu135*alu290*0.2316418882663604f)));
  var alu292 = (alu136*0.7071067811865475f);
  var alu293 = select(1.0f,-1.0f,(alu292<0.0f));
  var alu294 = select(0.0f,alu293,(alu292!=0.0f));
  var alu295 = (1/(1.0f+(alu136*alu294*0.2316418882663604f)));
  var alu296 = (alu137*0.7071067811865475f);
  var alu297 = select(1.0f,-1.0f,(alu296<0.0f));
  var alu298 = select(0.0f,alu297,(alu296!=0.0f));
  var alu299 = (1/(1.0f+(alu137*alu298*0.2316418882663604f)));
  var alu300 = (alu138*0.7071067811865475f);
  var alu301 = select(1.0f,-1.0f,(alu300<0.0f));
  var alu302 = select(0.0f,alu301,(alu300!=0.0f));
  var alu303 = (1/(1.0f+(alu138*alu302*0.2316418882663604f)));
  var alu304 = (alu139*0.7071067811865475f);
  var alu305 = select(1.0f,-1.0f,(alu304<0.0f));
  var alu306 = select(0.0f,alu305,(alu304!=0.0f));
  var alu307 = (1/(1.0f+(alu139*alu306*0.2316418882663604f)));
  var alu308 = (alu140*0.7071067811865475f);
  var alu309 = select(1.0f,-1.0f,(alu308<0.0f));
  var alu310 = select(0.0f,alu309,(alu308!=0.0f));
  var alu311 = (1/(1.0f+(alu140*alu310*0.2316418882663604f)));
  var alu312 = (alu141*0.7071067811865475f);
  var alu313 = select(1.0f,-1.0f,(alu312<0.0f));
  var alu314 = select(0.0f,alu313,(alu312!=0.0f));
  var alu315 = (1/(1.0f+(alu141*alu314*0.2316418882663604f)));
  var alu316 = (alu142*0.7071067811865475f);
  var alu317 = select(1.0f,-1.0f,(alu316<0.0f));
  var alu318 = select(0.0f,alu317,(alu316!=0.0f));
  var alu319 = (1/(1.0f+(alu142*alu318*0.2316418882663604f)));
  var alu320 = (alu143*0.7071067811865475f);
  var alu321 = select(1.0f,-1.0f,(alu320<0.0f));
  var alu322 = select(0.0f,alu321,(alu320!=0.0f));
  var alu323 = (1/(1.0f+(alu143*alu322*0.2316418882663604f)));
  var alu324 = (alu144*0.7071067811865475f);
  var alu325 = select(1.0f,-1.0f,(alu324<0.0f));
  var alu326 = select(0.0f,alu325,(alu324!=0.0f));
  var alu327 = (1/(1.0f+(alu144*alu326*0.2316418882663604f)));
  var alu328 = (alu145*0.7071067811865475f);
  var alu329 = select(1.0f,-1.0f,(alu328<0.0f));
  var alu330 = select(0.0f,alu329,(alu328!=0.0f));
  var alu331 = (1/(1.0f+(alu145*alu330*0.2316418882663604f)));
  var alu332 = (alu146*0.7071067811865475f);
  var alu333 = select(1.0f,-1.0f,(alu332<0.0f));
  var alu334 = select(0.0f,alu333,(alu332!=0.0f));
  var alu335 = (1/(1.0f+(alu146*alu334*0.2316418882663604f)));
  var alu336 = (alu147*0.7071067811865475f);
  var alu337 = select(1.0f,-1.0f,(alu336<0.0f));
  var alu338 = select(0.0f,alu337,(alu336!=0.0f));
  var alu339 = (1/(1.0f+(alu147*alu338*0.2316418882663604f)));
  data0_890880[(alu99+1)] = (alu112*(1.0f+(alu198*(1.0f-(alu199*((((((((1.061405429f*alu199)+-1.453152027f)*alu199)+1.421413741f)*alu199)+-0.284496736f)*alu199)+0.254829592f)*exp2((alu112*alu112*-0.7213475204444816f))))))*0.5f);
  data0_890880[(alu99+2)] = (alu124*(1.0f+(alu246*(1.0f-(alu247*((((((((1.061405429f*alu247)+-1.453152027f)*alu247)+1.421413741f)*alu247)+-0.284496736f)*alu247)+0.254829592f)*exp2((alu124*alu124*-0.7213475204444816f))))))*0.5f);
  data0_890880[(alu99+3)] = (alu136*(1.0f+(alu294*(1.0f-(alu295*((((((((1.061405429f*alu295)+-1.453152027f)*alu295)+1.421413741f)*alu295)+-0.284496736f)*alu295)+0.254829592f)*exp2((alu136*alu136*-0.7213475204444816f))))))*0.5f);
  data0_890880[(alu99+4)] = (alu103*(1.0f+(alu162*(1.0f-(alu163*((((((((1.061405429f*alu163)+-1.453152027f)*alu163)+1.421413741f)*alu163)+-0.284496736f)*alu163)+0.254829592f)*exp2((alu103*alu103*-0.7213475204444816f))))))*0.5f);
  data0_890880[(alu99+5)] = (alu115*(1.0f+(alu210*(1.0f-(alu211*((((((((1.061405429f*alu211)+-1.453152027f)*alu211)+1.421413741f)*alu211)+-0.284496736f)*alu211)+0.254829592f)*exp2((alu115*alu115*-0.7213475204444816f))))))*0.5f);
  data0_890880[(alu99+6)] = (alu127*(1.0f+(alu258*(1.0f-(alu259*((((((((1.061405429f*alu259)+-1.453152027f)*alu259)+1.421413741f)*alu259)+-0.284496736f)*alu259)+0.254829592f)*exp2((alu127*alu127*-0.7213475204444816f))))))*0.5f);
  data0_890880[(alu99+7)] = (alu139*(1.0f+(alu306*(1.0f-(alu307*((((((((1.061405429f*alu307)+-1.453152027f)*alu307)+1.421413741f)*alu307)+-0.284496736f)*alu307)+0.254829592f)*exp2((alu139*alu139*-0.7213475204444816f))))))*0.5f);
  data0_890880[(alu99+8)] = (alu106*(1.0f+(alu174*(1.0f-(alu175*((((((((1.061405429f*alu175)+-1.453152027f)*alu175)+1.421413741f)*alu175)+-0.284496736f)*alu175)+0.254829592f)*exp2((alu106*alu106*-0.7213475204444816f))))))*0.5f);
  data0_890880[(alu99+9)] = (alu118*(1.0f+(alu222*(1.0f-(alu223*((((((((1.061405429f*alu223)+-1.453152027f)*alu223)+1.421413741f)*alu223)+-0.284496736f)*alu223)+0.254829592f)*exp2((alu118*alu118*-0.7213475204444816f))))))*0.5f);
  data0_890880[(alu99+10)] = (alu130*(1.0f+(alu270*(1.0f-(alu271*((((((((1.061405429f*alu271)+-1.453152027f)*alu271)+1.421413741f)*alu271)+-0.284496736f)*alu271)+0.254829592f)*exp2((alu130*alu130*-0.7213475204444816f))))))*0.5f);
  data0_890880[(alu99+11)] = (alu142*(1.0f+(alu318*(1.0f-(alu319*((((((((1.061405429f*alu319)+-1.453152027f)*alu319)+1.421413741f)*alu319)+-0.284496736f)*alu319)+0.254829592f)*exp2((alu142*alu142*-0.7213475204444816f))))))*0.5f);
  data0_890880[(alu99+12)] = (alu109*(1.0f+(alu186*(1.0f-(alu187*((((((((1.061405429f*alu187)+-1.453152027f)*alu187)+1.421413741f)*alu187)+-0.284496736f)*alu187)+0.254829592f)*exp2((alu109*alu109*-0.7213475204444816f))))))*0.5f);
  data0_890880[(alu99+13)] = (alu121*(1.0f+(alu234*(1.0f-(alu235*((((((((1.061405429f*alu235)+-1.453152027f)*alu235)+1.421413741f)*alu235)+-0.284496736f)*alu235)+0.254829592f)*exp2((alu121*alu121*-0.7213475204444816f))))))*0.5f);
  data0_890880[(alu99+14)] = (alu133*(1.0f+(alu282*(1.0f-(alu283*((((((((1.061405429f*alu283)+-1.453152027f)*alu283)+1.421413741f)*alu283)+-0.284496736f)*alu283)+0.254829592f)*exp2((alu133*alu133*-0.7213475204444816f))))))*0.5f);
  data0_890880[(alu99+15)] = (alu145*(1.0f+(alu330*(1.0f-(alu331*((((((((1.061405429f*alu331)+-1.453152027f)*alu331)+1.421413741f)*alu331)+-0.284496736f)*alu331)+0.254829592f)*exp2((alu145*alu145*-0.7213475204444816f))))))*0.5f);
  data0_890880[(alu99+16)] = (alu101*(1.0f+(alu154*(1.0f-(alu155*((((((((1.061405429f*alu155)+-1.453152027f)*alu155)+1.421413741f)*alu155)+-0.284496736f)*alu155)+0.254829592f)*exp2((alu101*alu101*-0.7213475204444816f))))))*0.5f);
  data0_890880[(alu99+17)] = (alu113*(1.0f+(alu202*(1.0f-(alu203*((((((((1.061405429f*alu203)+-1.453152027f)*alu203)+1.421413741f)*alu203)+-0.284496736f)*alu203)+0.254829592f)*exp2((alu113*alu113*-0.7213475204444816f))))))*0.5f);
  data0_890880[(alu99+18)] = (alu125*(1.0f+(alu250*(1.0f-(alu251*((((((((1.061405429f*alu251)+-1.453152027f)*alu251)+1.421413741f)*alu251)+-0.284496736f)*alu251)+0.254829592f)*exp2((alu125*alu125*-0.7213475204444816f))))))*0.5f);
  data0_890880[(alu99+19)] = (alu137*(1.0f+(alu298*(1.0f-(alu299*((((((((1.061405429f*alu299)+-1.453152027f)*alu299)+1.421413741f)*alu299)+-0.284496736f)*alu299)+0.254829592f)*exp2((alu137*alu137*-0.7213475204444816f))))))*0.5f);
  data0_890880[(alu99+20)] = (alu104*(1.0f+(alu166*(1.0f-(alu167*((((((((1.061405429f*alu167)+-1.453152027f)*alu167)+1.421413741f)*alu167)+-0.284496736f)*alu167)+0.254829592f)*exp2((alu104*alu104*-0.7213475204444816f))))))*0.5f);
  data0_890880[(alu99+21)] = (alu116*(1.0f+(alu214*(1.0f-(alu215*((((((((1.061405429f*alu215)+-1.453152027f)*alu215)+1.421413741f)*alu215)+-0.284496736f)*alu215)+0.254829592f)*exp2((alu116*alu116*-0.7213475204444816f))))))*0.5f);
  data0_890880[(alu99+22)] = (alu128*(1.0f+(alu262*(1.0f-(alu263*((((((((1.061405429f*alu263)+-1.453152027f)*alu263)+1.421413741f)*alu263)+-0.284496736f)*alu263)+0.254829592f)*exp2((alu128*alu128*-0.7213475204444816f))))))*0.5f);
  data0_890880[(alu99+23)] = (alu140*(1.0f+(alu310*(1.0f-(alu311*((((((((1.061405429f*alu311)+-1.453152027f)*alu311)+1.421413741f)*alu311)+-0.284496736f)*alu311)+0.254829592f)*exp2((alu140*alu140*-0.7213475204444816f))))))*0.5f);
  data0_890880[(alu99+24)] = (alu107*(1.0f+(alu178*(1.0f-(alu179*((((((((1.061405429f*alu179)+-1.453152027f)*alu179)+1.421413741f)*alu179)+-0.284496736f)*alu179)+0.254829592f)*exp2((alu107*alu107*-0.7213475204444816f))))))*0.5f);
  data0_890880[(alu99+25)] = (alu119*(1.0f+(alu226*(1.0f-(alu227*((((((((1.061405429f*alu227)+-1.453152027f)*alu227)+1.421413741f)*alu227)+-0.284496736f)*alu227)+0.254829592f)*exp2((alu119*alu119*-0.7213475204444816f))))))*0.5f);
  data0_890880[(alu99+26)] = (alu131*(1.0f+(alu274*(1.0f-(alu275*((((((((1.061405429f*alu275)+-1.453152027f)*alu275)+1.421413741f)*alu275)+-0.284496736f)*alu275)+0.254829592f)*exp2((alu131*alu131*-0.7213475204444816f))))))*0.5f);
  data0_890880[(alu99+27)] = (alu143*(1.0f+(alu322*(1.0f-(alu323*((((((((1.061405429f*alu323)+-1.453152027f)*alu323)+1.421413741f)*alu323)+-0.284496736f)*alu323)+0.254829592f)*exp2((alu143*alu143*-0.7213475204444816f))))))*0.5f);
  data0_890880[(alu99+28)] = (alu110*(1.0f+(alu190*(1.0f-(alu191*((((((((1.061405429f*alu191)+-1.453152027f)*alu191)+1.421413741f)*alu191)+-0.284496736f)*alu191)+0.254829592f)*exp2((alu110*alu110*-0.7213475204444816f))))))*0.5f);
  data0_890880[(alu99+29)] = (alu122*(1.0f+(alu238*(1.0f-(alu239*((((((((1.061405429f*alu239)+-1.453152027f)*alu239)+1.421413741f)*alu239)+-0.284496736f)*alu239)+0.254829592f)*exp2((alu122*alu122*-0.7213475204444816f))))))*0.5f);
  data0_890880[(alu99+30)] = (alu134*(1.0f+(alu286*(1.0f-(alu287*((((((((1.061405429f*alu287)+-1.453152027f)*alu287)+1.421413741f)*alu287)+-0.284496736f)*alu287)+0.254829592f)*exp2((alu134*alu134*-0.7213475204444816f))))))*0.5f);
  data0_890880[(alu99+31)] = (alu146*(1.0f+(alu334*(1.0f-(alu335*((((((((1.061405429f*alu335)+-1.453152027f)*alu335)+1.421413741f)*alu335)+-0.284496736f)*alu335)+0.254829592f)*exp2((alu146*alu146*-0.7213475204444816f))))))*0.5f);
  data0_890880[(alu99+32)] = (alu102*(1.0f+(alu158*(1.0f-(alu159*((((((((1.061405429f*alu159)+-1.453152027f)*alu159)+1.421413741f)*alu159)+-0.284496736f)*alu159)+0.254829592f)*exp2((alu102*alu102*-0.7213475204444816f))))))*0.5f);
  data0_890880[(alu99+33)] = (alu114*(1.0f+(alu206*(1.0f-(alu207*((((((((1.061405429f*alu207)+-1.453152027f)*alu207)+1.421413741f)*alu207)+-0.284496736f)*alu207)+0.254829592f)*exp2((alu114*alu114*-0.7213475204444816f))))))*0.5f);
  data0_890880[(alu99+34)] = (alu126*(1.0f+(alu254*(1.0f-(alu255*((((((((1.061405429f*alu255)+-1.453152027f)*alu255)+1.421413741f)*alu255)+-0.284496736f)*alu255)+0.254829592f)*exp2((alu126*alu126*-0.7213475204444816f))))))*0.5f);
  data0_890880[(alu99+35)] = (alu138*(1.0f+(alu302*(1.0f-(alu303*((((((((1.061405429f*alu303)+-1.453152027f)*alu303)+1.421413741f)*alu303)+-0.284496736f)*alu303)+0.254829592f)*exp2((alu138*alu138*-0.7213475204444816f))))))*0.5f);
  data0_890880[(alu99+36)] = (alu105*(1.0f+(alu170*(1.0f-(alu171*((((((((1.061405429f*alu171)+-1.453152027f)*alu171)+1.421413741f)*alu171)+-0.284496736f)*alu171)+0.254829592f)*exp2((alu105*alu105*-0.7213475204444816f))))))*0.5f);
  data0_890880[(alu99+37)] = (alu117*(1.0f+(alu218*(1.0f-(alu219*((((((((1.061405429f*alu219)+-1.453152027f)*alu219)+1.421413741f)*alu219)+-0.284496736f)*alu219)+0.254829592f)*exp2((alu117*alu117*-0.7213475204444816f))))))*0.5f);
  data0_890880[(alu99+38)] = (alu129*(1.0f+(alu266*(1.0f-(alu267*((((((((1.061405429f*alu267)+-1.453152027f)*alu267)+1.421413741f)*alu267)+-0.284496736f)*alu267)+0.254829592f)*exp2((alu129*alu129*-0.7213475204444816f))))))*0.5f);
  data0_890880[(alu99+39)] = (alu141*(1.0f+(alu314*(1.0f-(alu315*((((((((1.061405429f*alu315)+-1.453152027f)*alu315)+1.421413741f)*alu315)+-0.284496736f)*alu315)+0.254829592f)*exp2((alu141*alu141*-0.7213475204444816f))))))*0.5f);
  data0_890880[(alu99+40)] = (alu108*(1.0f+(alu182*(1.0f-(alu183*((((((((1.061405429f*alu183)+-1.453152027f)*alu183)+1.421413741f)*alu183)+-0.284496736f)*alu183)+0.254829592f)*exp2((alu108*alu108*-0.7213475204444816f))))))*0.5f);
  data0_890880[(alu99+41)] = (alu120*(1.0f+(alu230*(1.0f-(alu231*((((((((1.061405429f*alu231)+-1.453152027f)*alu231)+1.421413741f)*alu231)+-0.284496736f)*alu231)+0.254829592f)*exp2((alu120*alu120*-0.7213475204444816f))))))*0.5f);
  data0_890880[(alu99+42)] = (alu132*(1.0f+(alu278*(1.0f-(alu279*((((((((1.061405429f*alu279)+-1.453152027f)*alu279)+1.421413741f)*alu279)+-0.284496736f)*alu279)+0.254829592f)*exp2((alu132*alu132*-0.7213475204444816f))))))*0.5f);
  data0_890880[(alu99+43)] = (alu144*(1.0f+(alu326*(1.0f-(alu327*((((((((1.061405429f*alu327)+-1.453152027f)*alu327)+1.421413741f)*alu327)+-0.284496736f)*alu327)+0.254829592f)*exp2((alu144*alu144*-0.7213475204444816f))))))*0.5f);
  data0_890880[(alu99+44)] = (alu111*(1.0f+(alu194*(1.0f-(alu195*((((((((1.061405429f*alu195)+-1.453152027f)*alu195)+1.421413741f)*alu195)+-0.284496736f)*alu195)+0.254829592f)*exp2((alu111*alu111*-0.7213475204444816f))))))*0.5f);
  data0_890880[(alu99+45)] = (alu123*(1.0f+(alu242*(1.0f-(alu243*((((((((1.061405429f*alu243)+-1.453152027f)*alu243)+1.421413741f)*alu243)+-0.284496736f)*alu243)+0.254829592f)*exp2((alu123*alu123*-0.7213475204444816f))))))*0.5f);
  data0_890880[(alu99+46)] = (alu135*(1.0f+(alu290*(1.0f-(alu291*((((((((1.061405429f*alu291)+-1.453152027f)*alu291)+1.421413741f)*alu291)+-0.284496736f)*alu291)+0.254829592f)*exp2((alu135*alu135*-0.7213475204444816f))))))*0.5f);
  data0_890880[(alu99+47)] = (alu147*(1.0f+(alu338*(1.0f-(alu339*((((((((1.061405429f*alu339)+-1.453152027f)*alu339)+1.421413741f)*alu339)+-0.284496736f)*alu339)+0.254829592f)*exp2((alu147*alu147*-0.7213475204444816f))))))*0.5f);
  data0_890880[alu99] = (alu100*(1.0f+(alu150*(1.0f-(alu151*((((((((1.061405429f*alu151)+-1.453152027f)*alu151)+1.421413741f)*alu151)+-0.284496736f)*alu151)+0.254829592f)*exp2((alu100*alu100*-0.7213475204444816f))))))*0.5f);
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

const r_10_32_29_4_3_2_384 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_222720:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_222720:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_147456:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_384:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_384:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_222720:array<f32>;
@compute @workgroup_size(29) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,24>;
  var gidx0 = i32(gindex.x); /* 32 */
  var gidx1 = i32(gindex.y); /* 10 */
  var lidx0 = i32(lindex.x); /* 29 */
  var alu0 = ((gidx1*22272)+(lidx0*384));
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
  for (var Ridx0 = 0; Ridx0 < 384; Ridx0++) {
    var alu25 = (alu0+Ridx0);
    var val0 = data1_222720[alu25];
    var alu26 = ((gidx0*4608)+Ridx0);
    var val1 = data2_147456[alu26];
    var val2 = data1_222720[(alu25+11136)];
    var val3 = data2_147456[(alu26+1536)];
    var val4 = data2_147456[(alu26+3072)];
    var val5 = data2_147456[(alu26+384)];
    var val6 = data2_147456[(alu26+1920)];
    var val7 = data2_147456[(alu26+3456)];
    var val8 = data2_147456[(alu26+768)];
    var val9 = data2_147456[(alu26+2304)];
    var val10 = data2_147456[(alu26+3840)];
    var val11 = data2_147456[(alu26+1152)];
    var val12 = data2_147456[(alu26+2688)];
    var val13 = data2_147456[(alu26+4224)];
    acc0[0] = (acc0[0]+(val0*val1));
    acc0[1] = (acc0[1]+(val2*val1));
    acc0[2] = (acc0[2]+(val0*val3));
    acc0[3] = (acc0[3]+(val2*val3));
    acc0[4] = (acc0[4]+(val0*val4));
    acc0[5] = (acc0[5]+(val2*val4));
    acc0[6] = (acc0[6]+(val0*val5));
    acc0[7] = (acc0[7]+(val2*val5));
    acc0[8] = (acc0[8]+(val0*val6));
    acc0[9] = (acc0[9]+(val2*val6));
    acc0[10] = (acc0[10]+(val0*val7));
    acc0[11] = (acc0[11]+(val2*val7));
    acc0[12] = (acc0[12]+(val0*val8));
    acc0[13] = (acc0[13]+(val2*val8));
    acc0[14] = (acc0[14]+(val0*val9));
    acc0[15] = (acc0[15]+(val2*val9));
    acc0[16] = (acc0[16]+(val0*val10));
    acc0[17] = (acc0[17]+(val2*val10));
    acc0[18] = (acc0[18]+(val0*val11));
    acc0[19] = (acc0[19]+(val2*val11));
    acc0[20] = (acc0[20]+(val0*val12));
    acc0[21] = (acc0[21]+(val2*val12));
    acc0[22] = (acc0[22]+(val0*val13));
    acc0[23] = (acc0[23]+(val2*val13));
  }
  var alu52 = (gidx0*12);
  var val14 = data3_384[alu52];
  var val15 = data4_384[alu52];
  var alu53 = (alu0+alu52);
  var alu54 = (alu53+1);
  var val16 = data5_222720[alu54];
  var val17 = data5_222720[alu53];
  var alu55 = (alu52+1);
  var val18 = data3_384[alu55];
  var val19 = data4_384[alu55];
  var alu56 = (alu52+2);
  var val20 = data3_384[alu56];
  var alu57 = (alu52+3);
  var val21 = data3_384[alu57];
  var val22 = data4_384[alu56];
  var alu58 = (alu53+2);
  var val23 = data5_222720[alu58];
  var val24 = data4_384[alu57];
  var alu59 = (alu53+3);
  var val25 = data5_222720[alu59];
  var alu60 = (alu52+4);
  var val26 = data3_384[alu60];
  var val27 = data4_384[alu60];
  var alu61 = (alu53+4);
  var val28 = data5_222720[alu61];
  var alu62 = (alu52+5);
  var val29 = data3_384[alu62];
  var val30 = data4_384[alu62];
  var alu63 = (alu53+5);
  var val31 = data5_222720[alu63];
  var alu64 = (alu52+6);
  var val32 = data3_384[alu64];
  var val33 = data4_384[alu64];
  var alu65 = (alu53+6);
  var val34 = data5_222720[alu65];
  var alu66 = (alu52+7);
  var val35 = data3_384[alu66];
  var alu67 = (alu52+8);
  var val36 = data3_384[alu67];
  var alu68 = (alu52+9);
  var val37 = data3_384[alu68];
  var val38 = data4_384[alu68];
  var alu69 = (alu53+9);
  var val39 = data5_222720[alu69];
  var alu70 = (alu52+10);
  var val40 = data3_384[alu70];
  var val41 = data4_384[alu70];
  var alu71 = (alu53+10);
  var val42 = data5_222720[alu71];
  var alu72 = (alu52+11);
  var val43 = data3_384[alu72];
  var val44 = data4_384[alu66];
  var alu73 = (alu53+7);
  var val45 = data5_222720[alu73];
  var val46 = data4_384[alu67];
  var alu74 = (alu53+8);
  var val47 = data5_222720[alu74];
  var val48 = data4_384[alu72];
  var alu75 = (alu53+11);
  var val49 = data5_222720[alu75];
  var alu76 = (alu53+11136);
  var val50 = data5_222720[alu76];
  var alu77 = (alu53+11137);
  var val51 = data5_222720[alu77];
  var alu78 = (alu53+11138);
  var val52 = data5_222720[alu78];
  var alu79 = (alu53+11139);
  var val53 = data5_222720[alu79];
  var alu80 = (alu53+11140);
  var val54 = data5_222720[alu80];
  var alu81 = (alu53+11141);
  var val55 = data5_222720[alu81];
  var alu82 = (alu53+11142);
  var val56 = data5_222720[alu82];
  var alu83 = (alu53+11143);
  var val57 = data5_222720[alu83];
  var alu84 = (alu53+11144);
  var val58 = data5_222720[alu84];
  var alu85 = (alu53+11145);
  var val59 = data5_222720[alu85];
  var alu86 = (alu53+11146);
  var val60 = data5_222720[alu86];
  var alu87 = (alu53+11147);
  var val61 = data5_222720[alu87];
  data0_222720[alu76] = (((acc0[1]+val14)*val15)+val50);
  data0_222720[alu77] = (((acc0[7]+val18)*val19)+val51);
  data0_222720[alu78] = (((acc0[13]+val20)*val22)+val52);
  data0_222720[alu79] = (((acc0[19]+val21)*val24)+val53);
  data0_222720[alu80] = (((acc0[3]+val26)*val27)+val54);
  data0_222720[alu81] = (((acc0[9]+val29)*val30)+val55);
  data0_222720[alu82] = (((acc0[15]+val32)*val33)+val56);
  data0_222720[alu83] = (((acc0[21]+val35)*val44)+val57);
  data0_222720[alu84] = (((acc0[5]+val36)*val46)+val58);
  data0_222720[alu85] = (((acc0[11]+val37)*val38)+val59);
  data0_222720[alu86] = (((acc0[17]+val40)*val41)+val60);
  data0_222720[alu87] = (((acc0[23]+val43)*val48)+val61);
  data0_222720[alu54] = (((acc0[6]+val18)*val19)+val16);
  data0_222720[alu58] = (((acc0[12]+val20)*val22)+val23);
  data0_222720[alu59] = (((acc0[18]+val21)*val24)+val25);
  data0_222720[alu61] = (((acc0[2]+val26)*val27)+val28);
  data0_222720[alu63] = (((acc0[8]+val29)*val30)+val31);
  data0_222720[alu65] = (((acc0[14]+val32)*val33)+val34);
  data0_222720[alu73] = (((acc0[20]+val35)*val44)+val45);
  data0_222720[alu74] = (((acc0[4]+val36)*val46)+val47);
  data0_222720[alu69] = (((acc0[10]+val37)*val38)+val39);
  data0_222720[alu71] = (((acc0[16]+val40)*val41)+val42);
  data0_222720[alu75] = (((acc0[22]+val43)*val48)+val49);
  data0_222720[alu53] = (((acc0[0]+val14)*val15)+val17);
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

const E_192_2_2_12_2_12 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_221184:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_222720:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_580:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_580:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_384:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_384:array<f32>;
@compute @workgroup_size(2) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 24 */
  var gidx1 = i32(gindex.y); /* 2 */
  var gidx2 = i32(gindex.z); /* 192 */
  var lidx0 = i32(lindex.x); /* 2 */
  var alu0 = (lidx0+bitcast<i32>((bitcast<u32>(gidx2)<<1u)));
  var alu1 = ((gidx0*11)>>7u);
  var alu2 = (gidx0-(12*alu1));
  var alu3 = (alu0+(alu2*384)+(gidx1*111360)+(alu1*55680));
  var val0 = data1_222720[(alu3+384)];
  var alu4 = ((gidx1*290)+(alu1*145)+alu2);
  var alu5 = (alu4+1);
  var val1 = data2_580[alu5];
  var val2 = data3_580[alu5];
  var val3 = data4_384[alu0];
  var val4 = data5_384[alu0];
  var val5 = data1_222720[(alu3+4992)];
  var alu6 = (alu4+13);
  var val6 = data2_580[alu6];
  var val7 = data3_580[alu6];
  var val8 = data1_222720[(alu3+9600)];
  var alu7 = (alu4+25);
  var val9 = data2_580[alu7];
  var val10 = data3_580[alu7];
  var val11 = data1_222720[(alu3+14208)];
  var alu8 = (alu4+37);
  var val12 = data2_580[alu8];
  var val13 = data3_580[alu8];
  var val14 = data1_222720[(alu3+18816)];
  var alu9 = (alu4+49);
  var val15 = data2_580[alu9];
  var val16 = data3_580[alu9];
  var val17 = data1_222720[(alu3+23424)];
  var alu10 = (alu4+61);
  var val18 = data2_580[alu10];
  var val19 = data3_580[alu10];
  var val20 = data1_222720[(alu3+28032)];
  var alu11 = (alu4+73);
  var val21 = data2_580[alu11];
  var val22 = data3_580[alu11];
  var val23 = data1_222720[(alu3+32640)];
  var alu12 = (alu4+85);
  var val24 = data2_580[alu12];
  var val25 = data3_580[alu12];
  var val26 = data1_222720[(alu3+37248)];
  var alu13 = (alu4+97);
  var val27 = data2_580[alu13];
  var alu14 = (alu4+121);
  var val28 = data2_580[alu14];
  var val29 = data3_580[alu13];
  var val30 = data1_222720[(alu3+41856)];
  var alu15 = (alu4+109);
  var val31 = data2_580[alu15];
  var val32 = data3_580[alu15];
  var val33 = data1_222720[(alu3+46464)];
  var val34 = data3_580[alu14];
  var val35 = data1_222720[(alu3+51072)];
  var alu16 = (alu4+133);
  var val36 = data2_580[alu16];
  var val37 = data3_580[alu16];
  var alu17 = (gidx0+(gidx1*288)+(gidx2*1152)+(lidx0*576));
  data0_221184[alu17] = (((val0-val1)*val2*val3)+val4);
  data0_221184[(alu17+24)] = (((val5-val6)*val7*val3)+val4);
  data0_221184[(alu17+48)] = (((val8-val9)*val10*val3)+val4);
  data0_221184[(alu17+72)] = (((val11-val12)*val13*val3)+val4);
  data0_221184[(alu17+96)] = (((val14-val15)*val16*val3)+val4);
  data0_221184[(alu17+120)] = (((val17-val18)*val19*val3)+val4);
  data0_221184[(alu17+144)] = (((val20-val21)*val22*val3)+val4);
  data0_221184[(alu17+168)] = (((val23-val24)*val25*val3)+val4);
  data0_221184[(alu17+192)] = (((val26-val27)*val29*val3)+val4);
  data0_221184[(alu17+216)] = (((val30-val31)*val32*val3)+val4);
  data0_221184[(alu17+240)] = (((val33-val28)*val34*val3)+val4);
  data0_221184[(alu17+264)] = (((val35-val36)*val37*val3)+val4);
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
    var alu6 = (alu5+(gidx1*1920));
    var val0 = data1_222720[alu6];
    var val1 = data2_222720[((gidx0*11136)+(lidx0*384)+alu5)];
    var val2 = data1_222720[(alu6+384)];
    var val3 = data1_222720[(alu6+768)];
    var val4 = data1_222720[(alu6+1152)];
    var val5 = data1_222720[(alu6+1536)];
    acc0[0] = (acc0[0]+(val0*val1));
    acc0[1] = (acc0[1]+(val2*val1));
    acc0[2] = (acc0[2]+(val3*val1));
    acc0[3] = (acc0[3]+(val4*val1));
    acc0[4] = (acc0[4]+(val5*val1));
  }
  var alu13 = (lidx0+(gidx0*29)+(gidx1*2900)+(gidx2*336400));
  data0_2018400[alu13] = (acc0[0]*0.125f);
  data0_2018400[(alu13+580)] = (acc0[1]*0.125f);
  data0_2018400[(alu13+1160)] = (acc0[2]*0.125f);
  data0_2018400[(alu13+1740)] = (acc0[3]*0.125f);
  data0_2018400[(alu13+2320)] = (acc0[4]*0.125f);
}`;

const r_120_29_145_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_3480:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_2018400:array<f32>;
@compute @workgroup_size(29) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 120 */
  var lidx0 = i32(lindex.x); /* 29 */
  acc0[0] = (f32(-INFINITY));
  for (var Ridx0 = 0; Ridx0 < 145; Ridx0++) {
    var alu1 = ((gidx0*16820)+(lidx0*580)+bitcast<i32>((bitcast<u32>(Ridx0)<<2u)));
    var val0 = data1_2018400[(alu1+1)];
    var val1 = data1_2018400[(alu1+2)];
    var val2 = data1_2018400[(alu1+3)];
    var val3 = data1_2018400[alu1];
    var alu2 = select(acc0[0],val3,(acc0[0]<val3));
    var alu3 = select(alu2,val0,(alu2<val0));
    var alu4 = select(alu3,val1,(alu3<val1));
    var alu5 = select(alu4,val2,(alu4<val2));
    acc0[0] = alu5;
  }
  data0_3480[(lidx0+(gidx0*29))] = acc0[0];
}`;

const r_120_29_145_4n1 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_3480:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_2018400:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_3480:array<f32>;
@compute @workgroup_size(29) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 120 */
  var lidx0 = i32(lindex.x); /* 29 */
  var alu0 = (lidx0+(gidx0*29));
  var val0 = data2_3480[alu0];
  acc0[0] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 145; Ridx0++) {
    var alu2 = ((gidx0*16820)+(lidx0*580)+bitcast<i32>((bitcast<u32>(Ridx0)<<2u)));
    var val1 = data1_2018400[(alu2+1)];
    var val2 = data1_2018400[(alu2+2)];
    var val3 = data1_2018400[(alu2+3)];
    var val4 = data1_2018400[alu2];
    acc0[0] = (acc0[0]+exp2(((val4-val0)*1.4426950408889634f))+exp2(((val1-val0)*1.4426950408889634f))+exp2(((val2-val0)*1.4426950408889634f))+exp2(((val3-val0)*1.4426950408889634f)));
  }
  data0_3480[alu0] = acc0[0];
}`;

const r_145_6_16_2_4_2_145_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_222720:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_2018400:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_3480:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_3480:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_222720:array<f32>;
@compute @workgroup_size(16,2) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,8>;
  var gidx0 = i32(gindex.x); /* 6 */
  var gidx1 = i32(gindex.y); /* 145 */
  var lidx1 = i32(lindex.y); /* 2 */
  var alu0 = (bitcast<i32>((bitcast<u32>(gidx1)<<2u))+bitcast<i32>((bitcast<u32>(lidx1)<<1u))+(gidx0*580));
  var val0 = data2_3480[alu0];
  var alu1 = (alu0+1);
  var val1 = data2_3480[alu1];
  var lidx0 = i32(lindex.x); /* 16 */
  var alu2 = (lidx0+bitcast<i32>((bitcast<u32>(gidx0)<<6u)));
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  acc0[4] = 0.0f;
  acc0[5] = 0.0f;
  acc0[6] = 0.0f;
  acc0[7] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 145; Ridx0++) {
    var alu11 = ((gidx1*2320)+(lidx1*1160)+bitcast<i32>((bitcast<u32>(Ridx0)<<2u))+(gidx0*336400));
    var val2 = data1_2018400[alu11];
    var alu12 = (alu2+(Ridx0*1536));
    var val3 = data4_222720[alu12];
    var val4 = data1_2018400[(alu11+1)];
    var val5 = data4_222720[(alu12+384)];
    var val6 = data1_2018400[(alu11+2)];
    var val7 = data4_222720[(alu12+768)];
    var val8 = data1_2018400[(alu11+3)];
    var val9 = data4_222720[(alu12+1152)];
    var val10 = data1_2018400[(alu11+580)];
    var val11 = data1_2018400[(alu11+581)];
    var val12 = data1_2018400[(alu11+582)];
    var val13 = data1_2018400[(alu11+583)];
    var val14 = data4_222720[(alu12+16)];
    var val15 = data4_222720[(alu12+32)];
    var val16 = data4_222720[(alu12+400)];
    var val17 = data4_222720[(alu12+784)];
    var val18 = data4_222720[(alu12+1168)];
    var val19 = data4_222720[(alu12+48)];
    var val20 = data4_222720[(alu12+416)];
    var val21 = data4_222720[(alu12+800)];
    var val22 = data4_222720[(alu12+1184)];
    var val23 = data4_222720[(alu12+432)];
    var val24 = data4_222720[(alu12+816)];
    var val25 = data4_222720[(alu12+1200)];
    var alu13 = exp2(((val4-val0)*1.4426950408889634f));
    var alu14 = exp2(((val6-val0)*1.4426950408889634f));
    var alu15 = exp2(((val8-val0)*1.4426950408889634f));
    var alu16 = exp2(((val10-val1)*1.4426950408889634f));
    var alu17 = exp2(((val11-val1)*1.4426950408889634f));
    var alu18 = exp2(((val12-val1)*1.4426950408889634f));
    var alu19 = exp2(((val13-val1)*1.4426950408889634f));
    var alu20 = exp2(((val2-val0)*1.4426950408889634f));
    acc0[0] = (acc0[0]+(alu20*val3)+(alu13*val5)+(alu14*val7)+(alu15*val9));
    acc0[1] = (acc0[1]+(alu16*val3)+(alu17*val5)+(alu18*val7)+(alu19*val9));
    acc0[2] = (acc0[2]+(alu20*val14)+(alu13*val16)+(alu14*val17)+(alu15*val18));
    acc0[3] = (acc0[3]+(alu16*val14)+(alu17*val16)+(alu18*val17)+(alu19*val18));
    acc0[4] = (acc0[4]+(alu20*val15)+(alu13*val20)+(alu14*val21)+(alu15*val22));
    acc0[5] = (acc0[5]+(alu16*val15)+(alu17*val20)+(alu18*val21)+(alu19*val22));
    acc0[6] = (acc0[6]+(alu20*val19)+(alu13*val23)+(alu14*val24)+(alu15*val25));
    acc0[7] = (acc0[7]+(alu16*val19)+(alu17*val23)+(alu18*val24)+(alu19*val25));
  }
  var val26 = data3_3480[alu0];
  var val27 = data3_3480[alu1];
  var alu30 = (alu2+(gidx1*1536)+(lidx1*768));
  var alu31 = (1/val26);
  var alu32 = (1/val27);
  data0_222720[alu30] = (acc0[0]*alu31);
  data0_222720[(alu30+16)] = (acc0[2]*alu31);
  data0_222720[(alu30+32)] = (acc0[4]*alu31);
  data0_222720[(alu30+48)] = (acc0[6]*alu31);
  data0_222720[(alu30+384)] = (acc0[1]*alu32);
  data0_222720[(alu30+400)] = (acc0[3]*alu32);
  data0_222720[(alu30+416)] = (acc0[5]*alu32);
  data0_222720[(alu30+432)] = (acc0[7]*alu32);
}`;

const r_10_32_29_4_3_2_384n1 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_222720:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_222720:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_147456:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_384:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_384:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_222720:array<f32>;
@compute @workgroup_size(29) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,24>;
  var gidx0 = i32(gindex.x); /* 32 */
  var gidx1 = i32(gindex.y); /* 10 */
  var lidx0 = i32(lindex.x); /* 29 */
  var alu0 = ((gidx1*22272)+(lidx0*384));
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
  for (var Ridx0 = 0; Ridx0 < 384; Ridx0++) {
    var alu25 = (alu0+Ridx0);
    var val0 = data1_222720[alu25];
    var alu26 = ((gidx0*4608)+Ridx0);
    var val1 = data2_147456[alu26];
    var val2 = data1_222720[(alu25+11136)];
    var val3 = data2_147456[(alu26+1536)];
    var val4 = data2_147456[(alu26+3072)];
    var val5 = data2_147456[(alu26+384)];
    var val6 = data2_147456[(alu26+1920)];
    var val7 = data2_147456[(alu26+3456)];
    var val8 = data2_147456[(alu26+768)];
    var val9 = data2_147456[(alu26+2304)];
    var val10 = data2_147456[(alu26+3840)];
    var val11 = data2_147456[(alu26+1152)];
    var val12 = data2_147456[(alu26+2688)];
    var val13 = data2_147456[(alu26+4224)];
    acc0[0] = (acc0[0]+(val0*val1));
    acc0[1] = (acc0[1]+(val2*val1));
    acc0[2] = (acc0[2]+(val0*val3));
    acc0[3] = (acc0[3]+(val2*val3));
    acc0[4] = (acc0[4]+(val0*val4));
    acc0[5] = (acc0[5]+(val2*val4));
    acc0[6] = (acc0[6]+(val0*val5));
    acc0[7] = (acc0[7]+(val2*val5));
    acc0[8] = (acc0[8]+(val0*val6));
    acc0[9] = (acc0[9]+(val2*val6));
    acc0[10] = (acc0[10]+(val0*val7));
    acc0[11] = (acc0[11]+(val2*val7));
    acc0[12] = (acc0[12]+(val0*val8));
    acc0[13] = (acc0[13]+(val2*val8));
    acc0[14] = (acc0[14]+(val0*val9));
    acc0[15] = (acc0[15]+(val2*val9));
    acc0[16] = (acc0[16]+(val0*val10));
    acc0[17] = (acc0[17]+(val2*val10));
    acc0[18] = (acc0[18]+(val0*val11));
    acc0[19] = (acc0[19]+(val2*val11));
    acc0[20] = (acc0[20]+(val0*val12));
    acc0[21] = (acc0[21]+(val2*val12));
    acc0[22] = (acc0[22]+(val0*val13));
    acc0[23] = (acc0[23]+(val2*val13));
  }
  var alu52 = (gidx0*12);
  var val14 = data3_384[alu52];
  var val15 = data4_384[alu52];
  var alu53 = (alu0+alu52);
  var alu54 = (alu53+1);
  var val16 = data5_222720[alu54];
  var val17 = data5_222720[alu53];
  var alu55 = (alu52+1);
  var val18 = data3_384[alu55];
  var val19 = data4_384[alu55];
  var alu56 = (alu52+2);
  var val20 = data3_384[alu56];
  var alu57 = (alu52+3);
  var val21 = data3_384[alu57];
  var val22 = data4_384[alu56];
  var alu58 = (alu53+2);
  var val23 = data5_222720[alu58];
  var val24 = data4_384[alu57];
  var alu59 = (alu53+3);
  var val25 = data5_222720[alu59];
  var alu60 = (alu52+4);
  var val26 = data3_384[alu60];
  var val27 = data4_384[alu60];
  var alu61 = (alu53+4);
  var val28 = data5_222720[alu61];
  var alu62 = (alu52+5);
  var val29 = data3_384[alu62];
  var val30 = data4_384[alu62];
  var alu63 = (alu53+5);
  var val31 = data5_222720[alu63];
  var alu64 = (alu52+6);
  var val32 = data3_384[alu64];
  var val33 = data4_384[alu64];
  var alu65 = (alu53+6);
  var val34 = data5_222720[alu65];
  var alu66 = (alu52+7);
  var val35 = data3_384[alu66];
  var alu67 = (alu52+8);
  var val36 = data3_384[alu67];
  var alu68 = (alu52+9);
  var val37 = data3_384[alu68];
  var val38 = data4_384[alu68];
  var alu69 = (alu53+9);
  var val39 = data5_222720[alu69];
  var alu70 = (alu52+10);
  var val40 = data3_384[alu70];
  var val41 = data4_384[alu70];
  var alu71 = (alu53+10);
  var val42 = data5_222720[alu71];
  var alu72 = (alu52+11);
  var val43 = data3_384[alu72];
  var val44 = data4_384[alu66];
  var alu73 = (alu53+7);
  var val45 = data5_222720[alu73];
  var val46 = data4_384[alu67];
  var alu74 = (alu53+8);
  var val47 = data5_222720[alu74];
  var val48 = data4_384[alu72];
  var alu75 = (alu53+11);
  var val49 = data5_222720[alu75];
  var alu76 = (alu53+11136);
  var val50 = data5_222720[alu76];
  var alu77 = (alu53+11137);
  var val51 = data5_222720[alu77];
  var alu78 = (alu53+11138);
  var val52 = data5_222720[alu78];
  var alu79 = (alu53+11139);
  var val53 = data5_222720[alu79];
  var alu80 = (alu53+11140);
  var val54 = data5_222720[alu80];
  var alu81 = (alu53+11141);
  var val55 = data5_222720[alu81];
  var alu82 = (alu53+11142);
  var val56 = data5_222720[alu82];
  var alu83 = (alu53+11143);
  var val57 = data5_222720[alu83];
  var alu84 = (alu53+11144);
  var val58 = data5_222720[alu84];
  var alu85 = (alu53+11145);
  var val59 = data5_222720[alu85];
  var alu86 = (alu53+11146);
  var val60 = data5_222720[alu86];
  var alu87 = (alu53+11147);
  var val61 = data5_222720[alu87];
  data0_222720[alu76] = (((acc0[1]+val14)*val15)+val50);
  data0_222720[alu77] = (((acc0[7]+val18)*val19)+val51);
  data0_222720[alu78] = (((acc0[13]+val20)*val22)+val52);
  data0_222720[alu79] = (((acc0[19]+val21)*val24)+val53);
  data0_222720[alu80] = (((acc0[3]+val26)*val27)+val54);
  data0_222720[alu81] = (((acc0[9]+val29)*val30)+val55);
  data0_222720[alu82] = (((acc0[15]+val32)*val33)+val56);
  data0_222720[alu83] = (((acc0[21]+val35)*val44)+val57);
  data0_222720[alu84] = (((acc0[5]+val36)*val46)+val58);
  data0_222720[alu85] = (((acc0[11]+val37)*val38)+val59);
  data0_222720[alu86] = (((acc0[17]+val40)*val41)+val60);
  data0_222720[alu87] = (((acc0[23]+val43)*val48)+val61);
  data0_222720[alu54] = (((acc0[6]+val18)*val19)+val16);
  data0_222720[alu58] = (((acc0[12]+val20)*val22)+val23);
  data0_222720[alu59] = (((acc0[18]+val21)*val24)+val25);
  data0_222720[alu61] = (((acc0[2]+val26)*val27)+val28);
  data0_222720[alu63] = (((acc0[8]+val29)*val30)+val31);
  data0_222720[alu65] = (((acc0[14]+val32)*val33)+val34);
  data0_222720[alu73] = (((acc0[20]+val35)*val44)+val45);
  data0_222720[alu74] = (((acc0[4]+val36)*val46)+val47);
  data0_222720[alu69] = (((acc0[10]+val37)*val38)+val39);
  data0_222720[alu71] = (((acc0[16]+val40)*val41)+val42);
  data0_222720[alu75] = (((acc0[22]+val43)*val48)+val49);
  data0_222720[alu53] = (((acc0[0]+val14)*val15)+val17);
}`;

const E_13824_32_2 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_884736:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_221184:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_221184:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_221184:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_221184:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 13824 */
  var lidx0 = i32(lindex.x); /* 32 */
  var alu0 = (lidx0+bitcast<i32>((bitcast<u32>(gidx0)<<6u)));
  var alu1 = (gidx0<3456);
  var val0 = select(0.0f, data1_221184[alu0], alu1);
  var alu2 = (alu0+32);
  var val1 = select(0.0f, data1_221184[alu2], alu1);
  var alu3 = ((3455<gidx0)&(gidx0<6912));
  var val2 = select(0.0f, data2_221184[(alu0+-221184)], alu3);
  var val3 = select(0.0f, data2_221184[(alu0+-221152)], alu3);
  var alu4 = ((6911<gidx0)&(gidx0<10368));
  var val4 = select(0.0f, data3_221184[(alu0+-442368)], alu4);
  var val5 = select(0.0f, data3_221184[(alu0+-442336)], alu4);
  var alu5 = (10367<gidx0);
  var val6 = select(0.0f, data4_221184[(alu0+-663552)], alu5);
  var val7 = select(0.0f, data4_221184[(alu0+-663520)], alu5);
  data0_884736[alu0] = (val0+val2+val4+val6);
  data0_884736[alu2] = (val1+val3+val5+val7);
}`;

const r_18_64_32_4_1536 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_147456:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_884736:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_393216:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 64 */
  var gidx1 = i32(gindex.y); /* 18 */
  var lidx0 = i32(lindex.x); /* 32 */
  var cast0 = bitcast<u32>(gidx1);
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 1536; Ridx0++) {
    var val0 = data1_884736[(lidx0+bitcast<i32>((cast0<<5u))+(Ridx0*576))];
    var alu4 = ((gidx0*6144)+Ridx0);
    var val1 = data2_393216[alu4];
    var val2 = data2_393216[(alu4+1536)];
    var val3 = data2_393216[(alu4+3072)];
    var val4 = data2_393216[(alu4+4608)];
    acc0[0] = (acc0[0]+(val0*val1));
    acc0[1] = (acc0[1]+(val0*val2));
    acc0[2] = (acc0[2]+(val0*val3));
    acc0[3] = (acc0[3]+(val0*val4));
  }
  var alu10 = (bitcast<i32>((cast0<<13u))+bitcast<i32>((bitcast<u32>(lidx0)<<8u))+bitcast<i32>((bitcast<u32>(gidx0)<<2u)));
  data0_147456[alu10] = acc0[0];
  data0_147456[(alu10+1)] = acc0[1];
  data0_147456[(alu10+2)] = acc0[2];
  data0_147456[(alu10+3)] = acc0[3];
}`;

const r_144_4_256 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_576:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_147456:array<f32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 144 */
  var cast0 = bitcast<u32>(gidx0);
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 256; Ridx0++) {
    var alu4 = (bitcast<i32>((cast0<<10u))+Ridx0);
    var val0 = data1_147456[alu4];
    var val1 = data1_147456[(alu4+256)];
    var val2 = data1_147456[(alu4+512)];
    var val3 = data1_147456[(alu4+768)];
    acc0[0] = (acc0[0]+val0);
    acc0[1] = (acc0[1]+val1);
    acc0[2] = (acc0[2]+val2);
    acc0[3] = (acc0[3]+val3);
  }
  var cast1 = bitcast<i32>((cast0<<2u));
  data0_576[cast1] = (acc0[0]*0.00390625f);
  data0_576[(cast1+1)] = (acc0[1]*0.00390625f);
  data0_576[(cast1+2)] = (acc0[2]*0.00390625f);
  data0_576[(cast1+3)] = (acc0[3]*0.00390625f);
}`;

const E_192_32_8_3 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_147456:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_147456:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_576:array<f32>;
@compute @workgroup_size(8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 32 */
  var gidx1 = i32(gindex.y); /* 192 */
  var lidx0 = i32(lindex.x); /* 8 */
  var alu0 = (lidx0+bitcast<i32>((bitcast<u32>(gidx0)<<3u))+(gidx1*768));
  var val0 = data1_147456[alu0];
  var alu1 = (gidx1*3);
  var val1 = data2_576[(alu1+1)];
  var val2 = data2_576[(alu1+2)];
  var val3 = data2_576[alu1];
  var alu2 = (alu0+256);
  var val4 = data1_147456[alu2];
  var alu3 = (alu0+512);
  var val5 = data1_147456[alu3];
  data0_147456[alu0] = (val0-val3);
  data0_147456[alu2] = (val4-val1);
  data0_147456[alu3] = (val5-val2);
}`;

const r_576_32_8 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32,32>;
@group(0) @binding(1)var<storage,read_write>data0_576:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_147456:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,1>;
  var acc1: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 576 */
  var lidx0 = i32(lindex.x); /* 32 */
  acc0[0] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 8; Ridx0++) {
    var val0 = data1_147456[(bitcast<i32>((bitcast<u32>(lidx0)<<3u))+Ridx0+bitcast<i32>((bitcast<u32>(gidx0)<<8u)))];
    acc0[0] = (acc0[0]+(val0*val0));
  }
  temp0[lidx0] = acc0[0];
  workgroupBarrier();
  acc1[0] = 0.0f;
  for (var Ridx102 = 0; Ridx102 < 32; Ridx102++) {
    var val1 = temp0[Ridx102];
    acc1[0] = (acc1[0]+val1);
  }
  var alu8 = (lidx0==0);
  if (alu8) {
    data0_576[gidx0] = sqrt(((acc1[0]*0.00390625f)+1e-06f));
  }
}`;

const E_256_24_8_3 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_147456:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_147456:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_576:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_256:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_256:array<f32>;
@compute @workgroup_size(8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 24 */
  var gidx1 = i32(gindex.y); /* 256 */
  var lidx0 = i32(lindex.x); /* 8 */
  var alu0 = (gidx1+(gidx0*6144)+bitcast<i32>((bitcast<u32>(lidx0)<<8u)));
  var val0 = data1_147456[alu0];
  var alu1 = (lidx0+(gidx0*24));
  var val1 = data2_576[alu1];
  var val2 = data3_256[gidx1];
  var val3 = data4_256[gidx1];
  var val4 = data1_147456[(alu0+2048)];
  var val5 = data2_576[(alu1+8)];
  var val6 = data1_147456[(alu0+4096)];
  var val7 = data2_576[(alu1+16)];
  var alu2 = (alu1+(gidx1*576));
  var alu3 = ((val0*(1/val1)*val2)+val3);
  var alu4 = ((val4*(1/val5)*val2)+val3);
  var alu5 = ((val6*(1/val7)*val2)+val3);
  data0_147456[alu2] = (alu3*(1/(1.0f+exp2((alu3*-1.4426950408889634f)))));
  data0_147456[(alu2+8)] = (alu4*(1/(1.0f+exp2((alu4*-1.4426950408889634f)))));
  data0_147456[(alu2+16)] = (alu5*(1/(1.0f+exp2((alu5*-1.4426950408889634f)))));
}`;

const r_6_6_16_4_4_8_128_3_3 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_73728:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_147456:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_147456:array<f32>;
@compute @workgroup_size(16,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,32>;
  var gidx0 = i32(gindex.x); /* 6 */
  var gidx1 = i32(gindex.y); /* 6 */
  var lidx0 = i32(lindex.x); /* 16 */
  var lidx1 = i32(lindex.y); /* 4 */
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
  for (var Ridx0 = 0; Ridx0 < 128; Ridx0++) {
    for (var Ridx2 = 0; Ridx2 < 3; Ridx2++) {
      var alu32 = (lidx1+bitcast<i32>((cast0<<2u))+Ridx2);
      var alu33 = (alu32+(gidx1*96)+(Ridx0*576));
      var alu34 = ((0<(gidx0+lidx1+Ridx2))&(alu32<25));
      var val0 = select(0.0f, data1_147456[(alu33+73703)], (alu34&(0<gidx1)));
      var val1 = select(0.0f, data1_147456[(alu33+73727)], alu34);
      var val2 = select(0.0f, data1_147456[(alu33+73751)], alu34);
      var val3 = select(0.0f, data1_147456[(alu33+73775)], alu34);
      var alu35 = ((Ridx0*9)+Ridx2+(lidx0*1152));
      var val4 = data2_147456[(alu35+3)];
      var val5 = data2_147456[(alu35+6)];
      var val6 = data2_147456[(alu35+18435)];
      var val7 = data2_147456[alu35];
      var val8 = data2_147456[(alu35+18432)];
      var val9 = data2_147456[(alu35+18438)];
      var val10 = data2_147456[(alu35+36864)];
      var val11 = data2_147456[(alu35+36867)];
      var val12 = data2_147456[(alu35+36870)];
      var val13 = data2_147456[(alu35+55296)];
      var val14 = data2_147456[(alu35+55299)];
      var val15 = data2_147456[(alu35+55302)];
      var val16 = data2_147456[(alu35+73728)];
      var val17 = data2_147456[(alu35+73731)];
      var val18 = data2_147456[(alu35+73734)];
      var val19 = data2_147456[(alu35+92160)];
      var val20 = data2_147456[(alu35+92163)];
      var val21 = data2_147456[(alu35+92166)];
      var val22 = data2_147456[(alu35+110592)];
      var val23 = data2_147456[(alu35+110595)];
      var val24 = data2_147456[(alu35+110598)];
      var val25 = data2_147456[(alu35+129024)];
      var val26 = data2_147456[(alu35+129027)];
      var val27 = data2_147456[(alu35+129030)];
      var val28 = select(0.0f, data1_147456[(alu33+73799)], alu34);
      var val29 = select(0.0f, data1_147456[(alu33+73823)], (alu34&(gidx1<5)));
      acc0[0] = (acc0[0]+(val0*val7)+(val1*val4)+(val2*val5));
      acc0[1] = (acc0[1]+(val0*val8)+(val1*val6)+(val2*val9));
      acc0[2] = (acc0[2]+(val0*val10)+(val1*val11)+(val2*val12));
      acc0[3] = (acc0[3]+(val0*val13)+(val1*val14)+(val2*val15));
      acc0[4] = (acc0[4]+(val0*val16)+(val1*val17)+(val2*val18));
      acc0[5] = (acc0[5]+(val0*val19)+(val1*val20)+(val2*val21));
      acc0[6] = (acc0[6]+(val0*val22)+(val1*val23)+(val2*val24));
      acc0[7] = (acc0[7]+(val0*val25)+(val1*val26)+(val2*val27));
      acc0[8] = (acc0[8]+(val1*val7)+(val2*val4)+(val3*val5));
      acc0[9] = (acc0[9]+(val1*val8)+(val2*val6)+(val3*val9));
      acc0[10] = (acc0[10]+(val1*val10)+(val2*val11)+(val3*val12));
      acc0[11] = (acc0[11]+(val1*val13)+(val2*val14)+(val3*val15));
      acc0[12] = (acc0[12]+(val1*val16)+(val2*val17)+(val3*val18));
      acc0[13] = (acc0[13]+(val1*val19)+(val2*val20)+(val3*val21));
      acc0[14] = (acc0[14]+(val1*val22)+(val2*val23)+(val3*val24));
      acc0[15] = (acc0[15]+(val1*val25)+(val2*val26)+(val3*val27));
      acc0[16] = (acc0[16]+(val2*val7)+(val3*val4)+(val28*val5));
      acc0[17] = (acc0[17]+(val2*val8)+(val3*val6)+(val28*val9));
      acc0[18] = (acc0[18]+(val2*val10)+(val3*val11)+(val28*val12));
      acc0[19] = (acc0[19]+(val2*val13)+(val3*val14)+(val28*val15));
      acc0[20] = (acc0[20]+(val2*val16)+(val3*val17)+(val28*val18));
      acc0[21] = (acc0[21]+(val2*val19)+(val3*val20)+(val28*val21));
      acc0[22] = (acc0[22]+(val2*val22)+(val3*val23)+(val28*val24));
      acc0[23] = (acc0[23]+(val2*val25)+(val3*val26)+(val28*val27));
      acc0[24] = (acc0[24]+(val3*val7)+(val28*val4)+(val29*val5));
      acc0[25] = (acc0[25]+(val3*val8)+(val28*val6)+(val29*val9));
      acc0[26] = (acc0[26]+(val3*val10)+(val28*val11)+(val29*val12));
      acc0[27] = (acc0[27]+(val3*val13)+(val28*val14)+(val29*val15));
      acc0[28] = (acc0[28]+(val3*val16)+(val28*val17)+(val29*val18));
      acc0[29] = (acc0[29]+(val3*val19)+(val28*val20)+(val29*val21));
      acc0[30] = (acc0[30]+(val3*val22)+(val28*val23)+(val29*val24));
      acc0[31] = (acc0[31]+(val3*val25)+(val28*val26)+(val29*val27));
    }
  }
  var alu70 = (lidx0+bitcast<i32>((cast0<<9u))+bitcast<i32>((bitcast<u32>(lidx1)<<7u))+(gidx1*12288));
  data0_73728[alu70] = acc0[0];
  data0_73728[(alu70+16)] = acc0[1];
  data0_73728[(alu70+32)] = acc0[2];
  data0_73728[(alu70+48)] = acc0[3];
  data0_73728[(alu70+64)] = acc0[4];
  data0_73728[(alu70+80)] = acc0[5];
  data0_73728[(alu70+96)] = acc0[6];
  data0_73728[(alu70+112)] = acc0[7];
  data0_73728[(alu70+3072)] = acc0[8];
  data0_73728[(alu70+3088)] = acc0[9];
  data0_73728[(alu70+3104)] = acc0[10];
  data0_73728[(alu70+3120)] = acc0[11];
  data0_73728[(alu70+3136)] = acc0[12];
  data0_73728[(alu70+3152)] = acc0[13];
  data0_73728[(alu70+3168)] = acc0[14];
  data0_73728[(alu70+3184)] = acc0[15];
  data0_73728[(alu70+6144)] = acc0[16];
  data0_73728[(alu70+6160)] = acc0[17];
  data0_73728[(alu70+6176)] = acc0[18];
  data0_73728[(alu70+6192)] = acc0[19];
  data0_73728[(alu70+6208)] = acc0[20];
  data0_73728[(alu70+6224)] = acc0[21];
  data0_73728[(alu70+6240)] = acc0[22];
  data0_73728[(alu70+6256)] = acc0[23];
  data0_73728[(alu70+9216)] = acc0[24];
  data0_73728[(alu70+9232)] = acc0[25];
  data0_73728[(alu70+9248)] = acc0[26];
  data0_73728[(alu70+9264)] = acc0[27];
  data0_73728[(alu70+9280)] = acc0[28];
  data0_73728[(alu70+9296)] = acc0[29];
  data0_73728[(alu70+9312)] = acc0[30];
  data0_73728[(alu70+9328)] = acc0[31];
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
    acc0[0] = (acc0[0]+val0+val1+val2+val3);
  }
  data0_576[gidx0] = (acc0[0]*0.0078125f);
}`;

const E_36_128_16 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_73728:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_73728:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_576:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 128 */
  var gidx1 = i32(gindex.y); /* 36 */
  var lidx0 = i32(lindex.x); /* 16 */
  var cast0 = bitcast<u32>(gidx1);
  var alu0 = (gidx0+bitcast<i32>((cast0<<11u))+bitcast<i32>((bitcast<u32>(lidx0)<<7u)));
  var val0 = data1_73728[alu0];
  var val1 = data2_576[(lidx0+bitcast<i32>((cast0<<4u)))];
  data0_73728[alu0] = (val0-val1);
}`;

const r_576_32_4n1 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
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

const r_12_8_16_2_4_6_128_3_3 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_73728:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_73728:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_147456:array<f32>;
@compute @workgroup_size(16,2) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,24>;
  var gidx0 = i32(gindex.x); /* 8 */
  var gidx1 = i32(gindex.y); /* 12 */
  var lidx0 = i32(lindex.x); /* 16 */
  var lidx1 = i32(lindex.y); /* 2 */
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
  for (var Ridx0 = 0; Ridx0 < 128; Ridx0++) {
    for (var Ridx1 = 0; Ridx1 < 3; Ridx1++) {
      var alu24 = ((gidx1*48)+(lidx1*24)+(Ridx1*24)+(Ridx0*576));
      var alu25 = ((0<(gidx1+lidx1+Ridx1))&((lidx1+bitcast<i32>((bitcast<u32>(gidx1)<<1u))+Ridx1)<25));
      var val0 = select(0.0f, data1_73728[(alu24+-24)], alu25);
      var alu26 = ((gidx0*18432)+(lidx0*1152)+(Ridx0*9)+(Ridx1*3));
      var val1 = data2_147456[(alu26+1)];
      var val2 = select(0.0f, data1_73728[(alu24+-23)], alu25);
      var val3 = data2_147456[(alu26+2)];
      var val4 = select(0.0f, data1_73728[(alu24+-21)], alu25);
      var val5 = data2_147456[alu26];
      var val6 = select(0.0f, data1_73728[(alu24+-20)], alu25);
      var val7 = select(0.0f, data1_73728[(alu24+-19)], alu25);
      var val8 = select(0.0f, data1_73728[(alu24+-17)], alu25);
      var val9 = select(0.0f, data1_73728[(alu24+-16)], alu25);
      var val10 = select(0.0f, data1_73728[(alu24+-15)], alu25);
      var val11 = select(0.0f, data1_73728[(alu24+-8)], alu25);
      var val12 = select(0.0f, data1_73728[(alu24+-7)], alu25);
      var val13 = select(0.0f, data1_73728[(alu24+-5)], alu25);
      var val14 = select(0.0f, data1_73728[(alu24+-4)], alu25);
      var val15 = select(0.0f, data1_73728[(alu24+-3)], alu25);
      var val16 = select(0.0f, data1_73728[(alu24+-22)], alu25);
      var val17 = select(0.0f, data1_73728[(alu24+-18)], alu25);
      var val18 = select(0.0f, data1_73728[(alu24+-14)], alu25);
      var val19 = select(0.0f, data1_73728[(alu24+-13)], alu25);
      var val20 = select(0.0f, data1_73728[(alu24+-12)], alu25);
      var val21 = select(0.0f, data1_73728[(alu24+-11)], alu25);
      var val22 = select(0.0f, data1_73728[(alu24+-9)], alu25);
      var val23 = select(0.0f, data1_73728[(alu24+-10)], alu25);
      var val24 = select(0.0f, data1_73728[(alu24+-6)], alu25);
      var val25 = select(0.0f, data1_73728[(alu24+-2)], alu25);
      var val26 = select(0.0f, data1_73728[(alu24+-1)], alu25);
      acc0[0] = (acc0[0]+(val0*val1)+(val2*val3));
      acc0[1] = (acc0[1]+(val4*val5)+(val6*val1)+(val7*val3));
      acc0[2] = (acc0[2]+(val8*val5)+(val9*val1)+(val10*val3));
      acc0[3] = (acc0[3]+(val19*val5)+(val20*val1)+(val21*val3));
      acc0[4] = (acc0[4]+(val22*val5)+(val11*val1)+(val12*val3));
      acc0[5] = (acc0[5]+(val13*val5)+(val14*val1)+(val15*val3));
      acc0[6] = (acc0[6]+(val0*val5)+(val2*val1)+(val16*val3));
      acc0[7] = (acc0[7]+(val6*val5)+(val7*val1)+(val17*val3));
      acc0[8] = (acc0[8]+(val9*val5)+(val10*val1)+(val18*val3));
      acc0[9] = (acc0[9]+(val20*val5)+(val21*val1)+(val23*val3));
      acc0[10] = (acc0[10]+(val11*val5)+(val12*val1)+(val24*val3));
      acc0[11] = (acc0[11]+(val14*val5)+(val15*val1)+(val25*val3));
      acc0[12] = (acc0[12]+(val2*val5)+(val16*val1)+(val4*val3));
      acc0[13] = (acc0[13]+(val7*val5)+(val17*val1)+(val8*val3));
      acc0[14] = (acc0[14]+(val10*val5)+(val18*val1)+(val19*val3));
      acc0[15] = (acc0[15]+(val21*val5)+(val23*val1)+(val22*val3));
      acc0[16] = (acc0[16]+(val12*val5)+(val24*val1)+(val13*val3));
      acc0[17] = (acc0[17]+(val15*val5)+(val25*val1)+(val26*val3));
      acc0[18] = (acc0[18]+(val16*val5)+(val4*val1)+(val6*val3));
      acc0[19] = (acc0[19]+(val17*val5)+(val8*val1)+(val9*val3));
      acc0[20] = (acc0[20]+(val18*val5)+(val19*val1)+(val20*val3));
      acc0[21] = (acc0[21]+(val23*val5)+(val22*val1)+(val11*val3));
      acc0[22] = (acc0[22]+(val24*val5)+(val13*val1)+(val14*val3));
      acc0[23] = (acc0[23]+(val25*val5)+(val26*val1));
    }
  }
  var alu53 = (lidx0+bitcast<i32>((bitcast<u32>(gidx0)<<4u))+(gidx1*6144)+(lidx1*3072));
  data0_73728[alu53] = acc0[0];
  data0_73728[(alu53+128)] = acc0[6];
  data0_73728[(alu53+256)] = acc0[12];
  data0_73728[(alu53+384)] = acc0[18];
  data0_73728[(alu53+512)] = acc0[1];
  data0_73728[(alu53+640)] = acc0[7];
  data0_73728[(alu53+768)] = acc0[13];
  data0_73728[(alu53+896)] = acc0[19];
  data0_73728[(alu53+1024)] = acc0[2];
  data0_73728[(alu53+1152)] = acc0[8];
  data0_73728[(alu53+1280)] = acc0[14];
  data0_73728[(alu53+1408)] = acc0[20];
  data0_73728[(alu53+1536)] = acc0[3];
  data0_73728[(alu53+1664)] = acc0[9];
  data0_73728[(alu53+1792)] = acc0[15];
  data0_73728[(alu53+1920)] = acc0[21];
  data0_73728[(alu53+2048)] = acc0[4];
  data0_73728[(alu53+2176)] = acc0[10];
  data0_73728[(alu53+2304)] = acc0[16];
  data0_73728[(alu53+2432)] = acc0[22];
  data0_73728[(alu53+2560)] = acc0[5];
  data0_73728[(alu53+2688)] = acc0[11];
  data0_73728[(alu53+2816)] = acc0[17];
  data0_73728[(alu53+2944)] = acc0[23];
}`;

const E_40_144_16_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
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
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 144 */
  var gidx1 = i32(gindex.y); /* 40 */
  var lidx0 = i32(lindex.x); /* 16 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<i32>((cast0<<2u));
  var alu0 = ((gidx1*9216)+(lidx0*576)+cast1);
  var alu1 = (gidx1<8);
  var val0 = select(0.0f, data1_147456[alu0], alu1);
  var alu2 = ((7<gidx1)&(gidx1<16));
  var val1 = select(0.0f, data1_147456[alu0], alu2);
  var alu3 = ((15<gidx1)&(gidx1<24));
  var val2 = select(0.0f, data2_73728[(alu0+-147456)], alu3);
  var alu4 = (gidx1<32);
  var alu5 = ((23<gidx1)&alu4);
  var val3 = select(0.0f, data3_73728[(alu0+-221184)], alu5);
  var alu6 = (lidx0+bitcast<i32>((bitcast<u32>(gidx1)<<4u)));
  var alu7 = (alu6+bitcast<i32>((cast0<<9u)));
  var alu8 = (31<gidx1);
  var val4 = select(0.0f, data4_73728[(alu7+-512)], alu8);
  var val5 = data5_576[cast1];
  var alu9 = (alu6+-512);
  var val6 = select(0.0f, data6_128[alu9], alu8);
  var val7 = select(0.0f, data7_128[alu9], alu8);
  var alu10 = (alu0+1);
  var val8 = select(0.0f, data1_147456[alu10], alu1);
  var val9 = select(0.0f, data1_147456[alu10], alu2);
  var val10 = select(0.0f, data2_73728[(alu0+-147455)], alu3);
  var val11 = select(0.0f, data3_73728[(alu0+-221183)], alu5);
  var val12 = select(0.0f, data4_73728[(alu7+-384)], alu8);
  var val13 = data5_576[(cast1+1)];
  var alu11 = (alu0+2);
  var val14 = select(0.0f, data1_147456[alu11], alu1);
  var val15 = select(0.0f, data1_147456[alu11], alu2);
  var val16 = select(0.0f, data2_73728[(alu0+-147454)], alu3);
  var val17 = select(0.0f, data3_73728[(alu0+-221182)], alu5);
  var val18 = select(0.0f, data4_73728[(alu7+-256)], alu8);
  var val19 = data5_576[(cast1+2)];
  var alu12 = (alu0+3);
  var val20 = select(0.0f, data1_147456[alu12], alu1);
  var val21 = select(0.0f, data1_147456[alu12], alu2);
  var val22 = select(0.0f, data2_73728[(alu0+-147453)], alu3);
  var val23 = select(0.0f, data3_73728[(alu0+-221181)], alu5);
  var val24 = select(0.0f, data4_73728[(alu7+-128)], alu8);
  var val25 = data5_576[(cast1+3)];
  var alu13 = ((val4*(1/val5)*val6)+val7);
  var alu14 = ((val12*(1/val13)*val6)+val7);
  var alu15 = ((val18*(1/val19)*val6)+val7);
  var alu16 = ((val24*(1/val25)*val6)+val7);
  var alu17 = select((alu14*(1/(1.0f+exp2((alu14*-1.4426950408889634f))))),0.0f,alu4);
  var alu18 = select((alu15*(1/(1.0f+exp2((alu15*-1.4426950408889634f))))),0.0f,alu4);
  var alu19 = select((alu16*(1/(1.0f+exp2((alu16*-1.4426950408889634f))))),0.0f,alu4);
  var alu20 = select((alu13*(1/(1.0f+exp2((alu13*-1.4426950408889634f))))),0.0f,alu4);
  data0_368640[alu10] = (val8+val9+val10+val11+alu17);
  data0_368640[alu11] = (val14+val15+val16+val17+alu18);
  data0_368640[alu12] = (val20+val21+val22+val23+alu19);
  data0_368640[alu0] = (val0+val1+val2+val3+alu20);
}`;

const r_18_256_32_160_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_147456:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_368640:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_163840:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 256 */
  var gidx1 = i32(gindex.y); /* 18 */
  var lidx0 = i32(lindex.x); /* 32 */
  var cast0 = bitcast<u32>(gidx1);
  acc0[0] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 160; Ridx0++) {
    var alu1 = (lidx0+bitcast<i32>((cast0<<5u))+(Ridx0*2304));
    var val0 = data1_368640[alu1];
    var alu2 = ((gidx0*640)+bitcast<i32>((bitcast<u32>(Ridx0)<<2u)));
    var val1 = data2_163840[alu2];
    var val2 = data1_368640[(alu1+576)];
    var val3 = data2_163840[(alu2+1)];
    var val4 = data1_368640[(alu1+1152)];
    var val5 = data2_163840[(alu2+2)];
    var val6 = data1_368640[(alu1+1728)];
    var val7 = data2_163840[(alu2+3)];
    acc0[0] = (acc0[0]+(val0*val1)+(val2*val3)+(val4*val5)+(val6*val7));
  }
  data0_147456[(gidx0+bitcast<i32>((cast0<<13u))+bitcast<i32>((bitcast<u32>(lidx0)<<8u)))] = acc0[0];
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

const E_72_64_8_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_147456:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_147456:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_576:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_256:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_256:array<f32>;
@compute @workgroup_size(8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 64 */
  var gidx1 = i32(gindex.y); /* 72 */
  var lidx0 = i32(lindex.x); /* 8 */
  var cast0 = bitcast<i32>((bitcast<u32>(gidx0)<<2u));
  var cast1 = bitcast<u32>(gidx1);
  var alu0 = (bitcast<i32>((cast1<<11u))+bitcast<i32>((bitcast<u32>(lidx0)<<8u))+cast0);
  var val0 = data1_147456[alu0];
  var val1 = data2_576[(lidx0+bitcast<i32>((cast1<<3u)))];
  var val2 = data3_256[cast0];
  var val3 = data4_256[cast0];
  var alu1 = (alu0+1);
  var val4 = data1_147456[alu1];
  var alu2 = (cast0+1);
  var val5 = data3_256[alu2];
  var alu3 = (cast0+2);
  var val6 = data3_256[alu3];
  var val7 = data4_256[alu2];
  var alu4 = (alu0+2);
  var val8 = data1_147456[alu4];
  var val9 = data4_256[alu3];
  var alu5 = (alu0+3);
  var val10 = data1_147456[alu5];
  var alu6 = (cast0+3);
  var val11 = data3_256[alu6];
  var val12 = data4_256[alu6];
  var alu7 = (1/val1);
  data0_147456[alu0] = ((val0*alu7*val2)+val3);
  data0_147456[alu1] = ((val4*alu7*val5)+val7);
  data0_147456[alu4] = ((val8*alu7*val6)+val9);
  data0_147456[alu5] = ((val10*alu7*val11)+val12);
}`;

const r_8_24_16_2_12_2_256 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_147456:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_147456:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_65536:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_256:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,48>;
  var gidx0 = i32(gindex.x); /* 24 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 16 */
  var cast0 = bitcast<u32>(gidx1);
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
  for (var Ridx2 = 0; Ridx2 < 256; Ridx2++) {
    var alu48 = (bitcast<i32>((bitcast<u32>(gidx0)<<8u))+Ridx2);
    var val0 = data1_147456[alu48];
    var alu49 = (bitcast<i32>((cast0<<13u))+bitcast<i32>((bitcast<u32>(lidx0)<<8u))+Ridx2);
    var val1 = data2_65536[alu49];
    var val2 = data2_65536[(alu49+4096)];
    var val3 = data1_147456[(alu48+12288)];
    var val4 = data1_147456[(alu48+24576)];
    var val5 = data1_147456[(alu48+36864)];
    var val6 = data1_147456[(alu48+49152)];
    var val7 = data1_147456[(alu48+61440)];
    var val8 = data1_147456[(alu48+73728)];
    var val9 = data1_147456[(alu48+86016)];
    var val10 = data1_147456[(alu48+98304)];
    var val11 = data1_147456[(alu48+110592)];
    var val12 = data1_147456[(alu48+135168)];
    var val13 = data1_147456[(alu48+6144)];
    var val14 = data1_147456[(alu48+18432)];
    var val15 = data1_147456[(alu48+30720)];
    var val16 = data1_147456[(alu48+43008)];
    var val17 = data1_147456[(alu48+55296)];
    var val18 = data1_147456[(alu48+67584)];
    var val19 = data1_147456[(alu48+79872)];
    var val20 = data1_147456[(alu48+122880)];
    var val21 = data1_147456[(alu48+92160)];
    var val22 = data1_147456[(alu48+104448)];
    var val23 = data1_147456[(alu48+116736)];
    var val24 = data1_147456[(alu48+129024)];
    var val25 = data1_147456[(alu48+141312)];
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
    acc0[10] = (acc0[10]+(val7*val1));
    acc0[11] = (acc0[11]+(val7*val2));
    acc0[12] = (acc0[12]+(val8*val1));
    acc0[13] = (acc0[13]+(val8*val2));
    acc0[14] = (acc0[14]+(val9*val1));
    acc0[15] = (acc0[15]+(val9*val2));
    acc0[16] = (acc0[16]+(val10*val1));
    acc0[17] = (acc0[17]+(val10*val2));
    acc0[18] = (acc0[18]+(val11*val1));
    acc0[19] = (acc0[19]+(val11*val2));
    acc0[20] = (acc0[20]+(val20*val1));
    acc0[21] = (acc0[21]+(val20*val2));
    acc0[22] = (acc0[22]+(val12*val1));
    acc0[23] = (acc0[23]+(val12*val2));
    acc0[24] = (acc0[24]+(val13*val1));
    acc0[25] = (acc0[25]+(val13*val2));
    acc0[26] = (acc0[26]+(val14*val1));
    acc0[27] = (acc0[27]+(val14*val2));
    acc0[28] = (acc0[28]+(val15*val1));
    acc0[29] = (acc0[29]+(val15*val2));
    acc0[30] = (acc0[30]+(val16*val1));
    acc0[31] = (acc0[31]+(val16*val2));
    acc0[32] = (acc0[32]+(val17*val1));
    acc0[33] = (acc0[33]+(val17*val2));
    acc0[34] = (acc0[34]+(val18*val1));
    acc0[35] = (acc0[35]+(val18*val2));
    acc0[36] = (acc0[36]+(val19*val1));
    acc0[37] = (acc0[37]+(val19*val2));
    acc0[38] = (acc0[38]+(val21*val1));
    acc0[39] = (acc0[39]+(val21*val2));
    acc0[40] = (acc0[40]+(val22*val1));
    acc0[41] = (acc0[41]+(val22*val2));
    acc0[42] = (acc0[42]+(val23*val1));
    acc0[43] = (acc0[43]+(val23*val2));
    acc0[44] = (acc0[44]+(val24*val1));
    acc0[45] = (acc0[45]+(val24*val2));
    acc0[46] = (acc0[46]+(val25*val1));
    acc0[47] = (acc0[47]+(val25*val2));
  }
  var alu99 = (lidx0+bitcast<i32>((cast0<<5u)));
  var val26 = data3_256[alu99];
  var val27 = data3_256[(alu99+16)];
  var alu100 = (gidx0+(gidx1*18432)+(lidx0*576));
  var alu101 = (16.0f*((f32((gidx0+1)))+-0.5f));
  var alu102 = select((alu101+-0.5f),0.0f,(alu101<0.5f));
  var alu103 = select(alu102,383.0f,(383.0f<alu102));
  var alu104 = trunc(alu103);
  var alu105 = select(alu104,(alu104+-1.0f),(alu103<alu104));
  var alu106 = (alu103-alu105);
  var alu107 = ((alu106+((alu106+(alu105-alu103))*0.5f))!=0.0f);
  var alu108 = select(0.0f,(acc0[0]+val26),alu107);
  var alu109 = select(0.0f,(acc0[1]+val27),alu107);
  var alu110 = select(0.0f,(acc0[2]+val26),alu107);
  var alu111 = select(0.0f,(acc0[3]+val27),alu107);
  var alu112 = select(0.0f,(acc0[4]+val26),alu107);
  var alu113 = select(0.0f,(acc0[5]+val27),alu107);
  var alu114 = select(0.0f,(acc0[6]+val26),alu107);
  var alu115 = select(0.0f,(acc0[7]+val27),alu107);
  var alu116 = select(0.0f,(acc0[8]+val26),alu107);
  var alu117 = select(0.0f,(acc0[9]+val27),alu107);
  var alu118 = select(0.0f,(acc0[10]+val26),alu107);
  var alu119 = select(0.0f,(acc0[11]+val27),alu107);
  var alu120 = select(0.0f,(acc0[12]+val26),alu107);
  var alu121 = select(0.0f,(acc0[13]+val27),alu107);
  var alu122 = select(0.0f,(acc0[14]+val26),alu107);
  var alu123 = select(0.0f,(acc0[15]+val27),alu107);
  var alu124 = select(0.0f,(acc0[16]+val26),alu107);
  var alu125 = select(0.0f,(acc0[17]+val27),alu107);
  var alu126 = select(0.0f,(acc0[18]+val26),alu107);
  var alu127 = select(0.0f,(acc0[19]+val27),alu107);
  var alu128 = select(0.0f,(acc0[20]+val26),alu107);
  var alu129 = select(0.0f,(acc0[21]+val27),alu107);
  var alu130 = select(0.0f,(acc0[22]+val26),alu107);
  var alu131 = select(0.0f,(acc0[23]+val27),alu107);
  var alu132 = select(0.0f,(acc0[24]+val26),alu107);
  var alu133 = select(0.0f,(acc0[25]+val27),alu107);
  var alu134 = select(0.0f,(acc0[26]+val26),alu107);
  var alu135 = select(0.0f,(acc0[27]+val27),alu107);
  var alu136 = select(0.0f,(acc0[28]+val26),alu107);
  var alu137 = select(0.0f,(acc0[29]+val27),alu107);
  var alu138 = select(0.0f,(acc0[30]+val26),alu107);
  var alu139 = select(0.0f,(acc0[31]+val27),alu107);
  var alu140 = select(0.0f,(acc0[32]+val26),alu107);
  var alu141 = select(0.0f,(acc0[33]+val27),alu107);
  var alu142 = select(0.0f,(acc0[34]+val26),alu107);
  var alu143 = select(0.0f,(acc0[35]+val27),alu107);
  var alu144 = select(0.0f,(acc0[36]+val26),alu107);
  var alu145 = select(0.0f,(acc0[37]+val27),alu107);
  var alu146 = select(0.0f,(acc0[38]+val26),alu107);
  var alu147 = select(0.0f,(acc0[39]+val27),alu107);
  var alu148 = select(0.0f,(acc0[40]+val26),alu107);
  var alu149 = select(0.0f,(acc0[41]+val27),alu107);
  var alu150 = select(0.0f,(acc0[42]+val26),alu107);
  var alu151 = select(0.0f,(acc0[43]+val27),alu107);
  var alu152 = select(0.0f,(acc0[44]+val26),alu107);
  var alu153 = select(0.0f,(acc0[45]+val27),alu107);
  var alu154 = select(0.0f,(acc0[46]+val26),alu107);
  var alu155 = select(0.0f,(acc0[47]+val27),alu107);
  data0_147456[alu100] = alu108;
  data0_147456[(alu100+24)] = alu132;
  data0_147456[(alu100+48)] = alu110;
  data0_147456[(alu100+72)] = alu134;
  data0_147456[(alu100+96)] = alu112;
  data0_147456[(alu100+120)] = alu136;
  data0_147456[(alu100+144)] = alu114;
  data0_147456[(alu100+168)] = alu138;
  data0_147456[(alu100+192)] = alu116;
  data0_147456[(alu100+216)] = alu140;
  data0_147456[(alu100+240)] = alu118;
  data0_147456[(alu100+264)] = alu142;
  data0_147456[(alu100+288)] = alu120;
  data0_147456[(alu100+312)] = alu144;
  data0_147456[(alu100+336)] = alu122;
  data0_147456[(alu100+360)] = alu146;
  data0_147456[(alu100+384)] = alu124;
  data0_147456[(alu100+408)] = alu148;
  data0_147456[(alu100+432)] = alu126;
  data0_147456[(alu100+456)] = alu150;
  data0_147456[(alu100+480)] = alu128;
  data0_147456[(alu100+504)] = alu152;
  data0_147456[(alu100+528)] = alu130;
  data0_147456[(alu100+552)] = alu154;
  data0_147456[(alu100+9216)] = alu109;
  data0_147456[(alu100+9240)] = alu133;
  data0_147456[(alu100+9264)] = alu111;
  data0_147456[(alu100+9288)] = alu135;
  data0_147456[(alu100+9312)] = alu113;
  data0_147456[(alu100+9336)] = alu137;
  data0_147456[(alu100+9360)] = alu115;
  data0_147456[(alu100+9384)] = alu139;
  data0_147456[(alu100+9408)] = alu117;
  data0_147456[(alu100+9432)] = alu141;
  data0_147456[(alu100+9456)] = alu119;
  data0_147456[(alu100+9480)] = alu143;
  data0_147456[(alu100+9504)] = alu121;
  data0_147456[(alu100+9528)] = alu145;
  data0_147456[(alu100+9552)] = alu123;
  data0_147456[(alu100+9576)] = alu147;
  data0_147456[(alu100+9600)] = alu125;
  data0_147456[(alu100+9624)] = alu149;
  data0_147456[(alu100+9648)] = alu127;
  data0_147456[(alu100+9672)] = alu151;
  data0_147456[(alu100+9696)] = alu129;
  data0_147456[(alu100+9720)] = alu153;
  data0_147456[(alu100+9744)] = alu131;
  data0_147456[(alu100+9768)] = alu155;
}`;

const r_12_16_8_4_2_3_4_24_24_4_64_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_147456:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_147456:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_65536:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_256:array<f32>;
@compute @workgroup_size(8,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<i32,1>;
  var acc1: array<i32,1>;
  var acc2: array<bool,6>;
  var acc3: array<f32,24>;
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
  var gidx1 = i32(gindex.y); /* 12 */
  var lidx0 = i32(lindex.x); /* 8 */
  var cast1 = bitcast<i32>((bitcast<u32>(gidx1)<<1u));
  var cast2 = (f32((cast1+1)));
  acc2[0] = false;
  acc2[1] = false;
  acc2[2] = false;
  acc2[3] = false;
  acc2[4] = false;
  acc2[5] = false;
  for (var Ridx4 = 0; Ridx4 < 4; Ridx4++) {
    var alu28 = (Ridx4<1);
    var alu29 = select(0,acc0[0],alu28);
    var alu30 = select(acc1[0],0,alu28);
    var alu31 = (1/(f32((alu29+alu30))));
    var alu32 = (Ridx4==1);
    var alu33 = select(0.0f,(f32(lidx0)),alu28);
    var alu34 = select(0.0f,(f32(cast1)),alu32);
    var alu35 = select(0.0f,cast2,alu32);
    var alu36 = select(0.0f,(f32((lidx0+8))),alu28);
    var alu37 = select(0.0f,(f32((lidx0+16))),alu28);
    var alu38 = (Ridx4<2);
    var alu39 = select(0.05f,((alu33+alu34+0.5f)*alu31),alu38);
    var alu40 = select(0.05f,((alu33+alu35+0.5f)*alu31),alu38);
    var alu41 = select(0.05f,((alu36+alu34+0.5f)*alu31),alu38);
    var alu42 = select(0.05f,((alu36+alu35+0.5f)*alu31),alu38);
    var alu43 = select(0.05f,((alu37+alu34+0.5f)*alu31),alu38);
    var alu44 = select(0.05f,((alu37+alu35+0.5f)*alu31),alu38);
    acc2[0] = (acc2[0]|(((0.01f<alu39)&(alu39<0.99f))!=true));
    acc2[1] = (acc2[1]|(((0.01f<alu41)&(alu41<0.99f))!=true));
    acc2[2] = (acc2[2]|(((0.01f<alu43)&(alu43<0.99f))!=true));
    acc2[3] = (acc2[3]|(((0.01f<alu40)&(alu40<0.99f))!=true));
    acc2[4] = (acc2[4]|(((0.01f<alu42)&(alu42<0.99f))!=true));
    acc2[5] = (acc2[5]|(((0.01f<alu44)&(alu44<0.99f))!=true));
  }
  var gidx0 = i32(gindex.x); /* 16 */
  var lidx1 = i32(lindex.y); /* 4 */
  var alu52 = (16.0f*(cast2+-0.5f));
  var alu53 = select((alu52+-0.5f),0.0f,(alu52<0.5f));
  var alu54 = select(alu53,383.0f,(383.0f<alu53));
  var alu55 = trunc(alu54);
  var cast3 = (i32(alu55));
  var alu56 = (16.0f*((f32((cast1+2)))+-0.5f));
  var alu57 = select((alu56+-0.5f),0.0f,(alu56<0.5f));
  var alu58 = select(alu57,383.0f,(383.0f<alu57));
  var alu59 = trunc(alu58);
  var cast4 = (i32(alu59));
  var alu60 = (alu55+-1.0f);
  var alu61 = (alu59+-1.0f);
  var cast5 = bitcast<u32>(gidx0);
  var alu62 = (16.0f*((f32((lidx0+1)))+-0.5f));
  var alu63 = select((alu62+-0.5f),0.0f,(alu62<0.5f));
  var alu64 = select(alu63,383.0f,(383.0f<alu63));
  var alu65 = trunc(alu64);
  var alu66 = (16.0f*((f32((lidx0+9)))+-0.5f));
  var alu67 = select((alu66+-0.5f),0.0f,(alu66<0.5f));
  var alu68 = select(alu67,383.0f,(383.0f<alu67));
  var alu69 = trunc(alu68);
  var alu70 = (16.0f*((f32((lidx0+17)))+-0.5f));
  var alu71 = select((alu70+-0.5f),0.0f,(alu70<0.5f));
  var alu72 = select(alu71,383.0f,(383.0f<alu71));
  var alu73 = trunc(alu72);
  var alu74 = ((gidx1*12288)+bitcast<i32>((bitcast<u32>(lidx0)<<8u)));
  var alu75 = select(cast3,(i32((alu55+1.0f))),(alu55<alu54));
  var alu76 = (alu54<alu55);
  var alu77 = select(cast3,(i32(alu60)),alu76);
  var alu78 = ((-1<alu75)&(alu75<384));
  var alu79 = ((-1<alu77)&(alu77<384));
  var alu80 = select(alu65,(alu65+-1.0f),(alu64<alu65));
  var alu81 = (alu64-alu80);
  var alu82 = select(alu55,alu60,alu76);
  var alu83 = (alu54-alu82);
  var alu84 = select(0.0f,alu81,alu78);
  var alu85 = select(0.0f,alu81,alu79);
  var alu86 = select(alu69,(alu69+-1.0f),(alu68<alu69));
  var alu87 = (alu68-alu86);
  var alu88 = select(0.0f,alu87,alu78);
  var alu89 = select(0.0f,alu87,alu79);
  var alu90 = select(alu73,(alu73+-1.0f),(alu72<alu73));
  var alu91 = (alu72-alu90);
  var alu92 = select(0.0f,alu91,alu78);
  var alu93 = select(0.0f,alu91,alu79);
  var alu94 = select(cast4,(i32((alu59+1.0f))),(alu59<alu58));
  var alu95 = (alu58<alu59);
  var alu96 = select(cast4,(i32(alu61)),alu95);
  var alu97 = ((-1<alu94)&(alu94<384));
  var alu98 = ((-1<alu96)&(alu96<384));
  var alu99 = select(alu59,alu61,alu95);
  var alu100 = (alu58-alu99);
  var alu101 = select(0.0f,alu81,alu97);
  var alu102 = select(0.0f,alu81,alu98);
  var alu103 = select(0.0f,alu87,alu97);
  var alu104 = select(0.0f,alu87,alu98);
  var alu105 = select(0.0f,alu91,alu97);
  var alu106 = select(0.0f,alu91,alu98);
  var alu107 = (((alu85+((alu84-alu85)*alu83))!=0.0f)&(acc2[0]!=true));
  var alu108 = (((alu89+((alu88-alu89)*alu83))!=0.0f)&(acc2[1]!=true));
  var alu109 = (((alu93+((alu92-alu93)*alu83))!=0.0f)&(acc2[2]!=true));
  var alu110 = (((alu102+((alu101-alu102)*alu100))!=0.0f)&(acc2[3]!=true));
  var alu111 = (((alu104+((alu103-alu104)*alu100))!=0.0f)&(acc2[4]!=true));
  var alu112 = (((alu106+((alu105-alu106)*alu100))!=0.0f)&(acc2[5]!=true));
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
  acc3[12] = 0.0f;
  acc3[13] = 0.0f;
  acc3[14] = 0.0f;
  acc3[15] = 0.0f;
  acc3[16] = 0.0f;
  acc3[17] = 0.0f;
  acc3[18] = 0.0f;
  acc3[19] = 0.0f;
  acc3[20] = 0.0f;
  acc3[21] = 0.0f;
  acc3[22] = 0.0f;
  acc3[23] = 0.0f;
  for (var Ridx5 = 0; Ridx5 < 64; Ridx5++) {
    var cast6 = bitcast<i32>((bitcast<u32>(Ridx5)<<2u));
    var alu137 = (bitcast<i32>((cast5<<12u))+bitcast<i32>((bitcast<u32>(lidx1)<<8u))+cast6);
    var val0 = data2_65536[alu137];
    var alu138 = (alu74+cast6);
    var val1 = select(0.0f, data1_147456[(alu138+1)], alu107);
    var val2 = data2_65536[(alu137+1)];
    var val3 = select(0.0f, data1_147456[(alu138+2)], alu107);
    var val4 = data2_65536[(alu137+2)];
    var val5 = select(0.0f, data1_147456[(alu138+3)], alu107);
    var val6 = select(0.0f, data1_147456[alu138], alu107);
    var val7 = data2_65536[(alu137+3)];
    var val8 = data2_65536[(alu137+1024)];
    var val9 = data2_65536[(alu137+1025)];
    var val10 = data2_65536[(alu137+1026)];
    var val11 = data2_65536[(alu137+1027)];
    var val12 = data2_65536[(alu137+2048)];
    var val13 = data2_65536[(alu137+2049)];
    var val14 = data2_65536[(alu137+2050)];
    var val15 = data2_65536[(alu137+2051)];
    var val16 = data2_65536[(alu137+3072)];
    var val17 = data2_65536[(alu137+3073)];
    var val18 = data2_65536[(alu137+3074)];
    var val19 = data2_65536[(alu137+3075)];
    var val20 = select(0.0f, data1_147456[(alu138+2048)], alu108);
    var val21 = select(0.0f, data1_147456[(alu138+2049)], alu108);
    var val22 = select(0.0f, data1_147456[(alu138+2050)], alu108);
    var val23 = select(0.0f, data1_147456[(alu138+2051)], alu108);
    var val24 = select(0.0f, data1_147456[(alu138+4096)], alu109);
    var val25 = select(0.0f, data1_147456[(alu138+4097)], alu109);
    var val26 = select(0.0f, data1_147456[(alu138+4098)], alu109);
    var val27 = select(0.0f, data1_147456[(alu138+4099)], alu109);
    var val28 = select(0.0f, data1_147456[(alu138+6144)], alu110);
    var val29 = select(0.0f, data1_147456[(alu138+6145)], alu110);
    var val30 = select(0.0f, data1_147456[(alu138+6146)], alu110);
    var val31 = select(0.0f, data1_147456[(alu138+6147)], alu110);
    var val32 = select(0.0f, data1_147456[(alu138+8192)], alu111);
    var val33 = select(0.0f, data1_147456[(alu138+8193)], alu111);
    var val34 = select(0.0f, data1_147456[(alu138+8194)], alu111);
    var val35 = select(0.0f, data1_147456[(alu138+8195)], alu111);
    var val36 = select(0.0f, data1_147456[(alu138+10240)], alu112);
    var val37 = select(0.0f, data1_147456[(alu138+10241)], alu112);
    var val38 = select(0.0f, data1_147456[(alu138+10242)], alu112);
    var val39 = select(0.0f, data1_147456[(alu138+10243)], alu112);
    acc3[0] = (acc3[0]+(val6*val0)+(val1*val2)+(val3*val4)+(val5*val7));
    acc3[1] = (acc3[1]+(val6*val8)+(val1*val9)+(val3*val10)+(val5*val11));
    acc3[2] = (acc3[2]+(val6*val12)+(val1*val13)+(val3*val14)+(val5*val15));
    acc3[3] = (acc3[3]+(val6*val16)+(val1*val17)+(val3*val18)+(val5*val19));
    acc3[4] = (acc3[4]+(val20*val0)+(val21*val2)+(val22*val4)+(val23*val7));
    acc3[5] = (acc3[5]+(val20*val8)+(val21*val9)+(val22*val10)+(val23*val11));
    acc3[6] = (acc3[6]+(val20*val12)+(val21*val13)+(val22*val14)+(val23*val15));
    acc3[7] = (acc3[7]+(val20*val16)+(val21*val17)+(val22*val18)+(val23*val19));
    acc3[8] = (acc3[8]+(val24*val0)+(val25*val2)+(val26*val4)+(val27*val7));
    acc3[9] = (acc3[9]+(val24*val8)+(val25*val9)+(val26*val10)+(val27*val11));
    acc3[10] = (acc3[10]+(val24*val12)+(val25*val13)+(val26*val14)+(val27*val15));
    acc3[11] = (acc3[11]+(val24*val16)+(val25*val17)+(val26*val18)+(val27*val19));
    acc3[12] = (acc3[12]+(val28*val0)+(val29*val2)+(val30*val4)+(val31*val7));
    acc3[13] = (acc3[13]+(val28*val8)+(val29*val9)+(val30*val10)+(val31*val11));
    acc3[14] = (acc3[14]+(val28*val12)+(val29*val13)+(val30*val14)+(val31*val15));
    acc3[15] = (acc3[15]+(val28*val16)+(val29*val17)+(val30*val18)+(val31*val19));
    acc3[16] = (acc3[16]+(val32*val0)+(val33*val2)+(val34*val4)+(val35*val7));
    acc3[17] = (acc3[17]+(val32*val8)+(val33*val9)+(val34*val10)+(val35*val11));
    acc3[18] = (acc3[18]+(val32*val12)+(val33*val13)+(val34*val14)+(val35*val15));
    acc3[19] = (acc3[19]+(val32*val16)+(val33*val17)+(val34*val18)+(val35*val19));
    acc3[20] = (acc3[20]+(val36*val0)+(val37*val2)+(val38*val4)+(val39*val7));
    acc3[21] = (acc3[21]+(val36*val8)+(val37*val9)+(val38*val10)+(val39*val11));
    acc3[22] = (acc3[22]+(val36*val12)+(val37*val13)+(val38*val14)+(val39*val15));
    acc3[23] = (acc3[23]+(val36*val16)+(val37*val17)+(val38*val18)+(val39*val19));
  }
  var alu164 = (lidx1+bitcast<i32>((cast5<<4u)));
  var val40 = data3_256[alu164];
  var val41 = data3_256[(alu164+4)];
  var val42 = data3_256[(alu164+8)];
  var val43 = data3_256[(alu164+12)];
  var alu165 = (alu164+alu74);
  data0_147456[alu165] = (acc3[0]+val40);
  data0_147456[(alu165+4)] = (acc3[1]+val41);
  data0_147456[(alu165+8)] = (acc3[2]+val42);
  data0_147456[(alu165+12)] = (acc3[3]+val43);
  data0_147456[(alu165+2048)] = (acc3[4]+val40);
  data0_147456[(alu165+2052)] = (acc3[5]+val41);
  data0_147456[(alu165+2056)] = (acc3[6]+val42);
  data0_147456[(alu165+2060)] = (acc3[7]+val43);
  data0_147456[(alu165+4096)] = (acc3[8]+val40);
  data0_147456[(alu165+4100)] = (acc3[9]+val41);
  data0_147456[(alu165+4104)] = (acc3[10]+val42);
  data0_147456[(alu165+4108)] = (acc3[11]+val43);
  data0_147456[(alu165+6144)] = (acc3[12]+val40);
  data0_147456[(alu165+6148)] = (acc3[13]+val41);
  data0_147456[(alu165+6152)] = (acc3[14]+val42);
  data0_147456[(alu165+6156)] = (acc3[15]+val43);
  data0_147456[(alu165+8192)] = (acc3[16]+val40);
  data0_147456[(alu165+8196)] = (acc3[17]+val41);
  data0_147456[(alu165+8200)] = (acc3[18]+val42);
  data0_147456[(alu165+8204)] = (acc3[19]+val43);
  data0_147456[(alu165+10240)] = (acc3[20]+val40);
  data0_147456[(alu165+10244)] = (acc3[21]+val41);
  data0_147456[(alu165+10248)] = (acc3[22]+val42);
  data0_147456[(alu165+10252)] = (acc3[23]+val43);
}`;

const r_144_4_256n1 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_576:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_147456:array<f32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 144 */
  var cast0 = bitcast<u32>(gidx0);
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 256; Ridx0++) {
    var alu4 = (bitcast<i32>((cast0<<10u))+Ridx0);
    var val0 = data1_147456[alu4];
    var val1 = data1_147456[(alu4+256)];
    var val2 = data1_147456[(alu4+512)];
    var val3 = data1_147456[(alu4+768)];
    acc0[0] = (acc0[0]+val0);
    acc0[1] = (acc0[1]+val1);
    acc0[2] = (acc0[2]+val2);
    acc0[3] = (acc0[3]+val3);
  }
  var cast1 = bitcast<i32>((cast0<<2u));
  data0_576[cast1] = (acc0[0]*0.00390625f);
  data0_576[(cast1+1)] = (acc0[1]*0.00390625f);
  data0_576[(cast1+2)] = (acc0[2]*0.00390625f);
  data0_576[(cast1+3)] = (acc0[3]*0.00390625f);
}`;

const r_576_16_16 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
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

const E_72_64_8_4n1 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
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

const r_96_91_3_2_256 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32,546>;
@group(0) @binding(1)var<storage,read_write>data0_576:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_147456:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_23296:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_91:array<f32>;
@compute @workgroup_size(91) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,6>;
  var acc1: array<f32,6>;
  var gidx0 = i32(gindex.x); /* 96 */
  var lidx0 = i32(lindex.x); /* 91 */
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  acc0[4] = 0.0f;
  acc0[5] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 256; Ridx0++) {
    var alu6 = ((gidx0*1536)+Ridx0);
    var val0 = data1_147456[alu6];
    var val1 = data2_23296[(bitcast<i32>((bitcast<u32>(lidx0)<<8u))+Ridx0)];
    var val2 = data1_147456[(alu6+768)];
    var val3 = data1_147456[(alu6+256)];
    var val4 = data1_147456[(alu6+512)];
    var val5 = data1_147456[(alu6+1024)];
    var val6 = data1_147456[(alu6+1280)];
    acc0[0] = (acc0[0]+(val0*val1));
    acc0[1] = (acc0[1]+(val2*val1));
    acc0[2] = (acc0[2]+(val3*val1));
    acc0[3] = (acc0[3]+(val5*val1));
    acc0[4] = (acc0[4]+(val4*val1));
    acc0[5] = (acc0[5]+(val6*val1));
  }
  var val7 = data3_91[lidx0];
  var alu14 = (lidx0*6);
  temp0[(alu14+1)] = (acc0[1]+val7);
  temp0[(alu14+2)] = (acc0[2]+val7);
  temp0[(alu14+3)] = (acc0[3]+val7);
  temp0[(alu14+4)] = (acc0[4]+val7);
  temp0[(alu14+5)] = (acc0[5]+val7);
  temp0[alu14] = (acc0[0]+val7);
  workgroupBarrier();
  acc1[0] = (f32(-INFINITY));
  acc1[1] = (f32(-INFINITY));
  acc1[2] = (f32(-INFINITY));
  acc1[3] = (f32(-INFINITY));
  acc1[4] = (f32(-INFINITY));
  acc1[5] = (f32(-INFINITY));
  for (var Ridx103 = 0; Ridx103 < 91; Ridx103++) {
    var alu28 = (Ridx103*6);
    var val8 = temp0[alu28];
    var val9 = temp0[(alu28+1)];
    var val10 = temp0[(alu28+2)];
    var val11 = temp0[(alu28+3)];
    var val12 = temp0[(alu28+4)];
    var val13 = temp0[(alu28+5)];
    var alu29 = select(acc1[0],val8,(acc1[0]<val8));
    var alu30 = select(acc1[1],val9,(acc1[1]<val9));
    var alu31 = select(acc1[2],val10,(acc1[2]<val10));
    var alu32 = select(acc1[3],val11,(acc1[3]<val11));
    var alu33 = select(acc1[4],val12,(acc1[4]<val12));
    var alu34 = select(acc1[5],val13,(acc1[5]<val13));
    acc1[0] = alu29;
    acc1[1] = alu30;
    acc1[2] = alu31;
    acc1[3] = alu32;
    acc1[4] = alu33;
    acc1[5] = alu34;
  }
  var alu42 = (gidx0*6);
  var alu43 = (lidx0==0);
  if (alu43) {
    data0_576[(alu42+1)] = acc1[2];
  }
  if (alu43) {
    data0_576[(alu42+2)] = acc1[4];
  }
  if (alu43) {
    data0_576[(alu42+3)] = acc1[1];
  }
  if (alu43) {
    data0_576[(alu42+4)] = acc1[3];
  }
  if (alu43) {
    data0_576[(alu42+5)] = acc1[5];
  }
  if (alu43) {
    data0_576[alu42] = acc1[0];
  }
}`;

const r_18_64_32_4_256 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_147456:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_147456:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_65536:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_256:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 64 */
  var gidx1 = i32(gindex.y); /* 18 */
  var lidx0 = i32(lindex.x); /* 32 */
  var cast0 = bitcast<u32>(gidx0);
  var alu0 = (bitcast<i32>((bitcast<u32>(gidx1)<<13u))+bitcast<i32>((bitcast<u32>(lidx0)<<8u)));
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 256; Ridx0++) {
    var val0 = data1_147456[(alu0+Ridx0)];
    var alu5 = (bitcast<i32>((cast0<<10u))+Ridx0);
    var val1 = data2_65536[alu5];
    var val2 = data2_65536[(alu5+256)];
    var val3 = data2_65536[(alu5+512)];
    var val4 = data2_65536[(alu5+768)];
    acc0[0] = (acc0[0]+(val0*val1));
    acc0[1] = (acc0[1]+(val0*val2));
    acc0[2] = (acc0[2]+(val0*val3));
    acc0[3] = (acc0[3]+(val0*val4));
  }
  var cast1 = bitcast<i32>((cast0<<2u));
  var val5 = data3_256[cast1];
  var val6 = data3_256[(cast1+1)];
  var val7 = data3_256[(cast1+2)];
  var val8 = data3_256[(cast1+3)];
  var alu11 = (alu0+cast1);
  var alu12 = (acc0[0]+val5);
  var alu13 = (acc0[1]+val6);
  var alu14 = (acc0[2]+val7);
  var alu15 = (acc0[3]+val8);
  var alu16 = select(0.0f,alu12,(0.0f<alu12));
  var alu17 = select(0.0f,alu13,(0.0f<alu13));
  var alu18 = select(0.0f,alu14,(0.0f<alu14));
  var alu19 = select(0.0f,alu15,(0.0f<alu15));
  data0_147456[alu11] = alu16;
  data0_147456[(alu11+1)] = alu17;
  data0_147456[(alu11+2)] = alu18;
  data0_147456[(alu11+3)] = alu19;
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

const r_576_16_36 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<i32,16>;
@group(0) @binding(1)var<storage,read_write>data0_576:array<i32>;
@group(0) @binding(2)var<storage,read_write>data1_576:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<i32,1>;
  var acc1: array<i32,1>;
  var gidx0 = i32(gindex.x); /* 576 */
  var val0 = data1_576[gidx0];
  var lidx0 = i32(lindex.x); /* 16 */
  acc0[0] = 0;
  for (var Ridx0 = 0; Ridx0 < 36; Ridx0++) {
    var alu1 = ((lidx0*36)+Ridx0);
    var val1 = data1_576[alu1];
    acc0[0] = (acc0[0]+(i32(((alu1<(gidx0+1))&(val1==val0)))));
  }
  temp0[lidx0] = acc0[0];
  workgroupBarrier();
  acc1[0] = 0;
  for (var Ridx102 = 0; Ridx102 < 16; Ridx102++) {
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

const r_36_4_16_256 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_2304:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_147456:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_1024:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_4:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 36 */
  var lidx0 = i32(lindex.x); /* 16 */
  var cast0 = bitcast<u32>(gidx1);
  var cast1 = bitcast<u32>(lidx0);
  acc0[0] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 256; Ridx0++) {
    var val0 = data1_147456[(bitcast<i32>((cast0<<12u))+bitcast<i32>((cast1<<8u))+Ridx0)];
    var val1 = data2_1024[(bitcast<i32>((bitcast<u32>(gidx0)<<8u))+Ridx0)];
    acc0[0] = (acc0[0]+(val0*val1));
  }
  var val2 = data3_4[gidx0];
  data0_2304[(gidx0+bitcast<i32>((cast0<<6u))+bitcast<i32>((cast1<<2u)))] = (acc0[0]+val2);
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

const r_576_16_36n1 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<i32,16>;
@group(0) @binding(1)var<storage,read_write>data0_576:array<i32>;
@group(0) @binding(2)var<storage,read_write>data1_1024:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<i32,1>;
  var acc1: array<i32,1>;
  var gidx0 = i32(gindex.x); /* 576 */
  var val0 = data1_1024[gidx0];
  var lidx0 = i32(lindex.x); /* 16 */
  acc0[0] = 0;
  for (var Ridx0 = 0; Ridx0 < 36; Ridx0++) {
    var alu1 = ((lidx0*36)+Ridx0);
    var val1 = data1_1024[alu1];
    acc0[0] = (acc0[0]+(i32(((alu1<(gidx0+1))&(val1==val0)))));
  }
  temp0[lidx0] = acc0[0];
  workgroupBarrier();
  acc1[0] = 0;
  for (var Ridx102 = 0; Ridx102 < 16; Ridx102++) {
    var val2 = temp0[Ridx102];
    acc1[0] = (acc1[0]+val2);
  }
  var alu9 = (lidx0==0);
  if (alu9) {
    data0_576[gidx0] = acc1[0];
  }
}`;

const r_576_16_36n2 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<i32,16>;
@group(0) @binding(1)var<storage,read_write>data0_576:array<i32>;
@group(0) @binding(2)var<storage,read_write>data1_576:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_1024:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_576:array<i32>;
@group(0) @binding(5)var<storage,read_write>data4_576:array<i32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<i32,1>;
  var acc1: array<i32,1>;
  var gidx0 = i32(gindex.x); /* 576 */
  var val0 = data4_576[gidx0];
  var val1 = data2_1024[gidx0];
  var lidx0 = i32(lindex.x); /* 16 */
  acc0[0] = 0;
  for (var Ridx0 = 0; Ridx0 < 36; Ridx0++) {
    var alu1 = ((lidx0*36)+Ridx0);
    var val2 = data3_576[alu1];
    var val3 = data1_576[alu1];
    acc0[0] = (acc0[0]+((i32(((val3==val1)&(val2==val0))))*alu1));
  }
  temp0[lidx0] = acc0[0];
  workgroupBarrier();
  acc1[0] = 0;
  for (var Ridx102 = 0; Ridx102 < 16; Ridx102++) {
    var val4 = temp0[Ridx102];
    acc1[0] = (acc1[0]+val4);
  }
  var alu9 = (lidx0==0);
  if (alu9) {
    data0_576[gidx0] = acc1[0];
  }
}`;

const r_75_4_24_4_24_24_24_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32,96>;
@group(0) @binding(1)var<storage,read_write>data0_1200:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_576:array<i32>;
@group(0) @binding(3)var<storage,read_write>data2_2304:array<f32>;
@compute @workgroup_size(24) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<i32,1>;
  var acc1: array<i32,1>;
  var acc2: array<f32,4>;
  var acc3: array<f32,4>;
  acc1[0] = 0;
  for (var Ridx2 = 0; Ridx2 < 24; Ridx2++) {
    var alu1 = (16.0f*((f32((Ridx2+1)))+-0.5f));
    var alu2 = select((alu1+-0.5f),0.0f,(alu1<0.5f));
    var alu3 = select(alu2,383.0f,(383.0f<alu2));
    var alu4 = trunc(alu3);
    var alu5 = select(alu4,(alu4+-1.0f),(alu3<alu4));
    acc1[0] = (acc1[0]+(i32(((alu3-alu5)!=0.0f))));
  }
  acc0[0] = 0;
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
    acc0[0] = (acc0[0]+(i32(((alu19+((alu18-alu19)*(alu11-alu17)))!=0.0f))));
  }
  var gidx1 = i32(gindex.y); /* 75 */
  var cast1 = bitcast<u32>(gidx1);
  var cast2 = bitcast<i32>((cast1<<2u));
  var val0 = data1_576[cast2];
  var val1 = data1_576[(cast2+1)];
  var val2 = data1_576[(cast2+2)];
  var val3 = data1_576[(cast2+3)];
  var gidx0 = i32(gindex.x); /* 4 */
  var lidx0 = i32(lindex.x); /* 24 */
  var alu22 = (16.0f*((f32((lidx0+1)))+-0.5f));
  var alu23 = select((alu22+-0.5f),0.0f,(alu22<0.5f));
  var alu24 = select(alu23,383.0f,(383.0f<alu23));
  var alu25 = trunc(alu24);
  var cast3 = (i32(alu25));
  var alu26 = (alu25+-1.0f);
  var cast4 = (f32(lidx0));
  var alu27 = (gidx0<1);
  var alu28 = select(0,acc1[0],alu27);
  var alu29 = select(acc0[0],0,alu27);
  var alu30 = ((cast4+0.5f)*(1/(f32(acc0[0]))));
  var alu31 = (gidx0<2);
  var alu32 = select(cast3,(i32((alu25+1.0f))),(alu25<alu24));
  var alu33 = (alu24<alu25);
  var alu34 = select(cast3,(i32(alu26)),alu33);
  var alu35 = select(alu25,alu26,alu33);
  var alu36 = select(0.0f,cast4,(gidx0==1));
  acc2[0] = 0.0f;
  acc2[1] = 0.0f;
  acc2[2] = 0.0f;
  acc2[3] = 0.0f;
  for (var Ridx5_1 = 0; Ridx5_1 < 24; Ridx5_1++) {
    var alu41 = (gidx0+(lidx0*96)+bitcast<i32>((bitcast<u32>(Ridx5_1)<<2u)));
    var val4 = select(0.0f, data2_2304[alu41], alu31);
    var val5 = select(0.0f, data2_2304[alu41], (1<gidx0));
    var cast5 = (f32(Ridx5_1));
    var alu42 = (16.0f*((f32((Ridx5_1+1)))+-0.5f));
    var alu43 = select((alu42+-0.5f),0.0f,(alu42<0.5f));
    var alu44 = select(alu43,383.0f,(383.0f<alu43));
    var alu45 = trunc(alu44);
    var alu46 = ((lidx0*24)+Ridx5_1);
    var alu47 = select(0.0f,cast5,alu27);
    var alu48 = select(alu45,(alu45+-1.0f),(alu44<alu45));
    var alu49 = (alu44-alu48);
    var alu50 = select(0.0f,alu49,((-1<alu32)&(alu32<384)));
    var alu51 = select(0.0f,alu49,((-1<alu34)&(alu34<384)));
    var alu52 = ((cast5+0.5f)*(1/(f32(acc1[0]))));
    var alu53 = ((alu51+((alu50-alu51)*(alu24-alu35)))!=0.0f);
    var alu54 = ((((0.01f<alu52)&(alu52<0.99f))!=true)|(((0.01f<alu30)&(alu30<0.99f))!=true));
    var alu55 = select(0.0f,0.05f,alu53);
    var alu56 = select(alu55,0.0f,alu54);
    var alu57 = select(0.05f,((alu47+alu36+0.5f)*(1/(f32((alu28+alu29))))),alu31);
    var alu58 = select(0.0f,alu57,alu53);
    var alu59 = select(alu58,0.0f,alu54);
    var alu60 = select(0.0f,((val4*alu56)+alu59),alu31);
    var alu61 = select((exp2((val5*1.4426950408889634f))*alu56),0.0f,alu31);
    var alu62 = (alu60+alu61);
    var alu63 = select(alu62,0.0f,(val0!=alu46));
    var alu64 = select(alu62,0.0f,(val1!=alu46));
    var alu65 = select(alu62,0.0f,(val2!=alu46));
    var alu66 = select(alu62,0.0f,(val3!=alu46));
    acc2[0] = (acc2[0]+alu63);
    acc2[1] = (acc2[1]+alu64);
    acc2[2] = (acc2[2]+alu65);
    acc2[3] = (acc2[3]+alu66);
  }
  var cast6 = bitcast<i32>((bitcast<u32>(lidx0)<<2u));
  temp0[cast6] = acc2[0];
  temp0[(cast6+1)] = acc2[1];
  temp0[(cast6+2)] = acc2[2];
  temp0[(cast6+3)] = acc2[3];
  workgroupBarrier();
  acc3[0] = 0.0f;
  acc3[1] = 0.0f;
  acc3[2] = 0.0f;
  acc3[3] = 0.0f;
  for (var Ridx110 = 0; Ridx110 < 24; Ridx110++) {
    var cast7 = bitcast<i32>((bitcast<u32>(Ridx110)<<2u));
    var val6 = temp0[cast7];
    var val7 = temp0[(cast7+1)];
    var val8 = temp0[(cast7+2)];
    var val9 = temp0[(cast7+3)];
    acc3[0] = (acc3[0]+val6);
    acc3[1] = (acc3[1]+val7);
    acc3[2] = (acc3[2]+val8);
    acc3[3] = (acc3[3]+val9);
  }
  var alu86 = (gidx0+bitcast<i32>((cast1<<4u)));
  var alu87 = (lidx0==0);
  if (alu87) {
    data0_1200[alu86] = acc3[0];
  }
  if (alu87) {
    data0_1200[(alu86+4)] = acc3[1];
  }
  if (alu87) {
    data0_1200[(alu86+8)] = acc3[2];
  }
  if (alu87) {
    data0_1200[(alu86+12)] = acc3[3];
  }
}`;

const r_60_64_32_4_5_8_2 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32,640>;
@group(0) @binding(1)var<storage,read_write>data0_76800:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_15600:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_1200:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_131072:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_256:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,20>;
  var acc1: array<f32,20>;
  var gidx1 = i32(gindex.y); /* 60 */
  var alu0 = (gidx1*20);
  var alu1 = (alu0+1);
  var val0 = data1_15600[alu1];
  var val1 = data2_1200[alu1];
  var alu2 = (alu0+3);
  var val2 = data2_1200[alu2];
  var alu3 = (alu0+2);
  var val3 = data1_15600[alu3];
  var val4 = data1_15600[alu0];
  var val5 = data2_1200[alu3];
  var val6 = data2_1200[alu0];
  var val7 = data1_15600[alu2];
  var alu4 = (alu0+4);
  var val8 = data1_15600[alu4];
  var alu5 = (alu0+5);
  var val9 = data1_15600[alu5];
  var val10 = data2_1200[alu4];
  var val11 = data2_1200[alu5];
  var alu6 = (alu0+6);
  var val12 = data2_1200[alu6];
  var alu7 = (alu0+7);
  var val13 = data2_1200[alu7];
  var val14 = data1_15600[alu6];
  var val15 = data1_15600[alu7];
  var alu8 = (alu0+9);
  var val16 = data1_15600[alu8];
  var val17 = data2_1200[alu8];
  var alu9 = (alu0+11);
  var val18 = data2_1200[alu9];
  var alu10 = (alu0+8);
  var val19 = data1_15600[alu10];
  var val20 = data2_1200[alu10];
  var alu11 = (alu0+10);
  var val21 = data2_1200[alu11];
  var val22 = data1_15600[alu11];
  var val23 = data1_15600[alu9];
  var alu12 = (alu0+13);
  var val24 = data1_15600[alu12];
  var alu13 = (alu0+12);
  var val25 = data2_1200[alu13];
  var val26 = data2_1200[alu12];
  var alu14 = (alu0+14);
  var val27 = data2_1200[alu14];
  var alu15 = (alu0+15);
  var val28 = data2_1200[alu15];
  var val29 = data1_15600[alu13];
  var val30 = data1_15600[alu14];
  var val31 = data1_15600[alu15];
  var alu16 = (alu0+17);
  var val32 = data1_15600[alu16];
  var val33 = data2_1200[alu16];
  var alu17 = (alu0+19);
  var val34 = data2_1200[alu17];
  var alu18 = (alu0+16);
  var val35 = data1_15600[alu18];
  var val36 = data2_1200[alu18];
  var alu19 = (alu0+18);
  var val37 = data2_1200[alu19];
  var val38 = data1_15600[alu19];
  var val39 = data1_15600[alu17];
  var gidx0 = i32(gindex.x); /* 64 */
  var lidx0 = i32(lindex.x); /* 32 */
  var cast0 = bitcast<u32>(gidx0);
  var alu20 = ((val0*val2)+val1);
  var alu21 = ((val8*val12)+val10);
  var alu22 = ((val9*val13)+val11);
  var alu23 = ((val19*val21)+val20);
  var alu24 = ((val16*val18)+val17);
  var alu25 = ((val29*val27)+val25);
  var alu26 = ((val24*val28)+val26);
  var alu27 = ((val35*val37)+val36);
  var alu28 = ((val32*val34)+val33);
  var alu29 = ((val4*val5)+val6);
  var alu30 = (exp2((val3*1.4426950408889634f))*val5);
  var alu31 = (exp2((val7*1.4426950408889634f))*val2);
  var alu32 = (exp2((val14*1.4426950408889634f))*val12);
  var alu33 = (exp2((val15*1.4426950408889634f))*val13);
  var alu34 = (exp2((val22*1.4426950408889634f))*val21);
  var alu35 = (exp2((val23*1.4426950408889634f))*val18);
  var alu36 = (exp2((val30*1.4426950408889634f))*val27);
  var alu37 = (exp2((val31*1.4426950408889634f))*val28);
  var alu38 = (exp2((val38*1.4426950408889634f))*val37);
  var alu39 = (exp2((val39*1.4426950408889634f))*val34);
  var alu40 = (lidx0<8);
  var alu41 = (lidx0<24);
  var alu42 = ((7<lidx0)&(lidx0<16));
  var alu43 = ((15<lidx0)&alu41);
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
  for (var Ridx0_0 = 0; Ridx0_0 < 8; Ridx0_0++) {
    var alu64 = (bitcast<i32>((bitcast<u32>(lidx0)<<4u))+bitcast<i32>((bitcast<u32>(Ridx0_0)<<1u)));
    var alu65 = (alu64+bitcast<i32>((cast0<<11u)));
    var val40 = data3_131072[alu65];
    var val41 = data3_131072[(alu65+1)];
    var val42 = data3_131072[(alu65+512)];
    var val43 = data3_131072[(alu65+513)];
    var val44 = data3_131072[(alu65+1024)];
    var val45 = data3_131072[(alu65+1025)];
    var val46 = data3_131072[(alu65+1536)];
    var val47 = data3_131072[(alu65+1537)];
    var alu66 = (((f32((alu64+-383)))+-1.0f)*0.5f);
    var alu67 = trunc(alu66);
    var alu68 = select(alu67,(alu67+-1.0f),(alu66<alu67));
    var alu69 = (((f32((alu64+-382)))+-1.0f)*0.5f);
    var alu70 = trunc(alu69);
    var alu71 = select(alu70,(alu70+-1.0f),(alu69<alu70));
    var alu72 = (((f32((alu64+-255)))+-1.0f)*0.5f);
    var alu73 = trunc(alu72);
    var alu74 = select(alu73,(alu73+-1.0f),(alu72<alu73));
    var alu75 = (((f32((alu64+-254)))+-1.0f)*0.5f);
    var alu76 = trunc(alu75);
    var alu77 = select(alu76,(alu76+-1.0f),(alu75<alu76));
    var alu78 = (((f32((alu64+-127)))+-1.0f)*0.5f);
    var alu79 = trunc(alu78);
    var alu80 = select(alu79,(alu79+-1.0f),(alu78<alu79));
    var alu81 = (((f32((alu64+-126)))+-1.0f)*0.5f);
    var alu82 = trunc(alu81);
    var alu83 = select(alu82,(alu82+-1.0f),(alu81<alu82));
    var alu84 = (((f32((alu64+1)))+-1.0f)*0.5f);
    var alu85 = trunc(alu84);
    var alu86 = select(alu85,(alu85+-1.0f),(alu84<alu85));
    var alu87 = (((f32((alu64+2)))+-1.0f)*0.5f);
    var alu88 = trunc(alu87);
    var alu89 = select(alu88,(alu88+-1.0f),(alu87<alu88));
    var alu90 = (1/exp2((alu86*0.20762050593046014f)));
    var alu91 = (1/exp2((alu80*0.20762050593046014f)));
    var alu92 = (1/exp2((alu74*0.20762050593046014f)));
    var alu93 = (1/exp2((alu68*0.20762050593046014f)));
    var alu94 = (1/exp2((alu89*0.20762050593046014f)));
    var alu95 = select(0.0f,(alu20*alu94*6.283185307179586f),alu40);
    var alu96 = select(0.0f,(alu22*alu94*6.283185307179586f),alu40);
    var alu97 = select(0.0f,(alu24*alu94*6.283185307179586f),alu40);
    var alu98 = select(0.0f,(alu26*alu94*6.283185307179586f),alu40);
    var alu99 = select(0.0f,(alu28*alu94*6.283185307179586f),alu40);
    var alu100 = (1/exp2((alu71*0.20762050593046014f)));
    var alu101 = select((alu31*alu100*6.283185307179586f),0.0f,alu41);
    var alu102 = select((alu33*alu100*6.283185307179586f),0.0f,alu41);
    var alu103 = select((alu35*alu100*6.283185307179586f),0.0f,alu41);
    var alu104 = select((alu37*alu100*6.283185307179586f),0.0f,alu41);
    var alu105 = select((alu39*alu100*6.283185307179586f),0.0f,alu41);
    var alu106 = (1/exp2((alu83*0.20762050593046014f)));
    var alu107 = select(0.0f,(alu21*alu106*6.283185307179586f),alu42);
    var alu108 = select(0.0f,(alu23*alu106*6.283185307179586f),alu42);
    var alu109 = select(0.0f,(alu25*alu106*6.283185307179586f),alu42);
    var alu110 = select(0.0f,(alu27*alu106*6.283185307179586f),alu42);
    var alu111 = select(0.0f,(alu29*alu106*6.283185307179586f),alu42);
    var alu112 = (1/exp2((alu77*0.20762050593046014f)));
    var alu113 = select(0.0f,(alu30*alu112*6.283185307179586f),alu43);
    var alu114 = select(0.0f,(alu32*alu112*6.283185307179586f),alu43);
    var alu115 = select(0.0f,(alu34*alu112*6.283185307179586f),alu43);
    var alu116 = select(0.0f,(alu36*alu112*6.283185307179586f),alu43);
    var alu117 = select(0.0f,(alu38*alu112*6.283185307179586f),alu43);
    var alu118 = select(0.0f,sin((alu20*alu90*6.283185307179586f)),alu40);
    var alu119 = select(0.0f,sin((alu29*alu91*6.283185307179586f)),alu42);
    var alu120 = select(0.0f,sin((alu30*alu92*6.283185307179586f)),alu43);
    var alu121 = select(sin((alu31*alu93*6.283185307179586f)),0.0f,alu41);
    var alu122 = select(alu121,0.0f,alu41);
    var alu123 = (alu118+alu119+alu120+alu122);
    var alu124 = select(0.0f,sin((alu22*alu90*6.283185307179586f)),alu40);
    var alu125 = select(0.0f,sin((alu21*alu91*6.283185307179586f)),alu42);
    var alu126 = select(0.0f,sin((alu32*alu92*6.283185307179586f)),alu43);
    var alu127 = select(sin((alu33*alu93*6.283185307179586f)),0.0f,alu41);
    var alu128 = select(alu127,0.0f,alu41);
    var alu129 = (alu124+alu125+alu126+alu128);
    var alu130 = select(0.0f,sin((alu24*alu90*6.283185307179586f)),alu40);
    var alu131 = select(0.0f,sin((alu23*alu91*6.283185307179586f)),alu42);
    var alu132 = select(0.0f,sin((alu34*alu92*6.283185307179586f)),alu43);
    var alu133 = select(sin((alu35*alu93*6.283185307179586f)),0.0f,alu41);
    var alu134 = select(alu133,0.0f,alu41);
    var alu135 = (alu130+alu131+alu132+alu134);
    var alu136 = select(0.0f,sin((alu26*alu90*6.283185307179586f)),alu40);
    var alu137 = select(0.0f,sin((alu25*alu91*6.283185307179586f)),alu42);
    var alu138 = select(0.0f,sin((alu36*alu92*6.283185307179586f)),alu43);
    var alu139 = select(sin((alu37*alu93*6.283185307179586f)),0.0f,alu41);
    var alu140 = select(alu139,0.0f,alu41);
    var alu141 = (alu136+alu137+alu138+alu140);
    var alu142 = select(0.0f,sin((alu28*alu90*6.283185307179586f)),alu40);
    var alu143 = select(0.0f,sin((alu27*alu91*6.283185307179586f)),alu42);
    var alu144 = select(0.0f,sin((alu38*alu92*6.283185307179586f)),alu43);
    var alu145 = select(sin((alu39*alu93*6.283185307179586f)),0.0f,alu41);
    var alu146 = select(alu145,0.0f,alu41);
    var alu147 = (alu142+alu143+alu144+alu146);
    var alu148 = select(0.0f,sin((1.5707963267948966f-alu95)),alu40);
    var alu149 = select(0.0f,sin((1.5707963267948966f-alu111)),alu42);
    var alu150 = select(0.0f,sin((1.5707963267948966f-alu113)),alu43);
    var alu151 = select(sin((1.5707963267948966f-alu101)),0.0f,alu41);
    var alu152 = select(alu151,0.0f,alu41);
    var alu153 = (alu148+alu149+alu150+alu152);
    var alu154 = select(0.0f,sin((1.5707963267948966f-alu96)),alu40);
    var alu155 = select(0.0f,sin((1.5707963267948966f-alu107)),alu42);
    var alu156 = select(0.0f,sin((1.5707963267948966f-alu114)),alu43);
    var alu157 = select(sin((1.5707963267948966f-alu102)),0.0f,alu41);
    var alu158 = select(alu157,0.0f,alu41);
    var alu159 = (alu154+alu155+alu156+alu158);
    var alu160 = select(0.0f,sin((1.5707963267948966f-alu97)),alu40);
    var alu161 = select(0.0f,sin((1.5707963267948966f-alu108)),alu42);
    var alu162 = select(0.0f,sin((1.5707963267948966f-alu115)),alu43);
    var alu163 = select(sin((1.5707963267948966f-alu103)),0.0f,alu41);
    var alu164 = select(alu163,0.0f,alu41);
    var alu165 = (alu160+alu161+alu162+alu164);
    var alu166 = select(0.0f,sin((1.5707963267948966f-alu98)),alu40);
    var alu167 = select(0.0f,sin((1.5707963267948966f-alu109)),alu42);
    var alu168 = select(0.0f,sin((1.5707963267948966f-alu116)),alu43);
    var alu169 = select(sin((1.5707963267948966f-alu104)),0.0f,alu41);
    var alu170 = select(alu169,0.0f,alu41);
    var alu171 = (alu166+alu167+alu168+alu170);
    var alu172 = select(0.0f,sin((1.5707963267948966f-alu99)),alu40);
    var alu173 = select(0.0f,sin((1.5707963267948966f-alu110)),alu42);
    var alu174 = select(0.0f,sin((1.5707963267948966f-alu117)),alu43);
    var alu175 = select(sin((1.5707963267948966f-alu105)),0.0f,alu41);
    var alu176 = select(alu175,0.0f,alu41);
    var alu177 = (alu172+alu173+alu174+alu176);
    acc0[0] = (acc0[0]+(alu123*val40)+(alu153*val41));
    acc0[1] = (acc0[1]+(alu129*val40)+(alu159*val41));
    acc0[2] = (acc0[2]+(alu135*val40)+(alu165*val41));
    acc0[3] = (acc0[3]+(alu141*val40)+(alu171*val41));
    acc0[4] = (acc0[4]+(alu147*val40)+(alu177*val41));
    acc0[5] = (acc0[5]+(alu123*val42)+(alu153*val43));
    acc0[6] = (acc0[6]+(alu129*val42)+(alu159*val43));
    acc0[7] = (acc0[7]+(alu135*val42)+(alu165*val43));
    acc0[8] = (acc0[8]+(alu141*val42)+(alu171*val43));
    acc0[9] = (acc0[9]+(alu147*val42)+(alu177*val43));
    acc0[10] = (acc0[10]+(alu123*val44)+(alu153*val45));
    acc0[11] = (acc0[11]+(alu129*val44)+(alu159*val45));
    acc0[12] = (acc0[12]+(alu135*val44)+(alu165*val45));
    acc0[13] = (acc0[13]+(alu141*val44)+(alu171*val45));
    acc0[14] = (acc0[14]+(alu147*val44)+(alu177*val45));
    acc0[15] = (acc0[15]+(alu123*val46)+(alu153*val47));
    acc0[16] = (acc0[16]+(alu129*val46)+(alu159*val47));
    acc0[17] = (acc0[17]+(alu135*val46)+(alu165*val47));
    acc0[18] = (acc0[18]+(alu141*val46)+(alu171*val47));
    acc0[19] = (acc0[19]+(alu147*val46)+(alu177*val47));
  }
  var alu199 = (lidx0*20);
  temp0[(alu199+1)] = acc0[1];
  temp0[(alu199+2)] = acc0[2];
  temp0[(alu199+3)] = acc0[3];
  temp0[(alu199+4)] = acc0[4];
  temp0[(alu199+5)] = acc0[5];
  temp0[(alu199+6)] = acc0[6];
  temp0[(alu199+7)] = acc0[7];
  temp0[(alu199+8)] = acc0[8];
  temp0[(alu199+9)] = acc0[9];
  temp0[(alu199+10)] = acc0[10];
  temp0[(alu199+11)] = acc0[11];
  temp0[(alu199+12)] = acc0[12];
  temp0[(alu199+13)] = acc0[13];
  temp0[(alu199+14)] = acc0[14];
  temp0[(alu199+15)] = acc0[15];
  temp0[(alu199+16)] = acc0[16];
  temp0[(alu199+17)] = acc0[17];
  temp0[(alu199+18)] = acc0[18];
  temp0[(alu199+19)] = acc0[19];
  temp0[alu199] = acc0[0];
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
  acc1[16] = 0.0f;
  acc1[17] = 0.0f;
  acc1[18] = 0.0f;
  acc1[19] = 0.0f;
  for (var Ridx103 = 0; Ridx103 < 32; Ridx103++) {
    var alu241 = (Ridx103*20);
    var val48 = temp0[alu241];
    var val49 = temp0[(alu241+1)];
    var val50 = temp0[(alu241+2)];
    var val51 = temp0[(alu241+3)];
    var val52 = temp0[(alu241+4)];
    var val53 = temp0[(alu241+5)];
    var val54 = temp0[(alu241+6)];
    var val55 = temp0[(alu241+7)];
    var val56 = temp0[(alu241+8)];
    var val57 = temp0[(alu241+9)];
    var val58 = temp0[(alu241+10)];
    var val59 = temp0[(alu241+11)];
    var val60 = temp0[(alu241+12)];
    var val61 = temp0[(alu241+13)];
    var val62 = temp0[(alu241+14)];
    var val63 = temp0[(alu241+15)];
    var val64 = temp0[(alu241+16)];
    var val65 = temp0[(alu241+17)];
    var val66 = temp0[(alu241+18)];
    var val67 = temp0[(alu241+19)];
    acc1[0] = (acc1[0]+val48);
    acc1[1] = (acc1[1]+val49);
    acc1[2] = (acc1[2]+val50);
    acc1[3] = (acc1[3]+val51);
    acc1[4] = (acc1[4]+val52);
    acc1[5] = (acc1[5]+val53);
    acc1[6] = (acc1[6]+val54);
    acc1[7] = (acc1[7]+val55);
    acc1[8] = (acc1[8]+val56);
    acc1[9] = (acc1[9]+val57);
    acc1[10] = (acc1[10]+val58);
    acc1[11] = (acc1[11]+val59);
    acc1[12] = (acc1[12]+val60);
    acc1[13] = (acc1[13]+val61);
    acc1[14] = (acc1[14]+val62);
    acc1[15] = (acc1[15]+val63);
    acc1[16] = (acc1[16]+val64);
    acc1[17] = (acc1[17]+val65);
    acc1[18] = (acc1[18]+val66);
    acc1[19] = (acc1[19]+val67);
  }
  var cast1 = bitcast<i32>((cast0<<2u));
  var val68 = data4_256[cast1];
  var val69 = data4_256[(cast1+1)];
  var val70 = data4_256[(cast1+2)];
  var val71 = data4_256[(cast1+3)];
  var alu263 = (cast1+(gidx1*1280));
  var alu264 = (lidx0==0);
  var alu265 = (acc1[0]+val68);
  var alu266 = (acc1[5]+val69);
  var alu267 = (acc1[10]+val70);
  var alu268 = (acc1[15]+val71);
  var alu269 = select(0.0f,alu265,(0.0f<alu265));
  var alu270 = select(0.0f,alu266,(0.0f<alu266));
  var alu271 = select(0.0f,alu267,(0.0f<alu267));
  var alu272 = select(0.0f,alu268,(0.0f<alu268));
  if (alu264) {
    data0_76800[alu263] = alu269;
  }
  if (alu264) {
    data0_76800[(alu263+1)] = alu270;
  }
  if (alu264) {
    data0_76800[(alu263+2)] = alu271;
  }
  if (alu264) {
    data0_76800[(alu263+3)] = alu272;
  }
  var alu285 = (acc1[1]+val68);
  var alu286 = (acc1[6]+val69);
  var alu287 = (acc1[11]+val70);
  var alu288 = (acc1[16]+val71);
  var alu289 = select(0.0f,alu285,(0.0f<alu285));
  var alu290 = select(0.0f,alu286,(0.0f<alu286));
  var alu291 = select(0.0f,alu287,(0.0f<alu287));
  var alu292 = select(0.0f,alu288,(0.0f<alu288));
  if (alu264) {
    data0_76800[(alu263+256)] = alu289;
  }
  if (alu264) {
    data0_76800[(alu263+257)] = alu290;
  }
  if (alu264) {
    data0_76800[(alu263+258)] = alu291;
  }
  if (alu264) {
    data0_76800[(alu263+259)] = alu292;
  }
  var alu305 = (acc1[2]+val68);
  var alu306 = (acc1[7]+val69);
  var alu307 = (acc1[12]+val70);
  var alu308 = (acc1[17]+val71);
  var alu309 = select(0.0f,alu305,(0.0f<alu305));
  var alu310 = select(0.0f,alu306,(0.0f<alu306));
  var alu311 = select(0.0f,alu307,(0.0f<alu307));
  var alu312 = select(0.0f,alu308,(0.0f<alu308));
  if (alu264) {
    data0_76800[(alu263+512)] = alu309;
  }
  if (alu264) {
    data0_76800[(alu263+513)] = alu310;
  }
  if (alu264) {
    data0_76800[(alu263+514)] = alu311;
  }
  if (alu264) {
    data0_76800[(alu263+515)] = alu312;
  }
  var alu325 = (acc1[3]+val68);
  var alu326 = (acc1[8]+val69);
  var alu327 = (acc1[13]+val70);
  var alu328 = (acc1[18]+val71);
  var alu329 = select(0.0f,alu325,(0.0f<alu325));
  var alu330 = select(0.0f,alu326,(0.0f<alu326));
  var alu331 = select(0.0f,alu327,(0.0f<alu327));
  var alu332 = select(0.0f,alu328,(0.0f<alu328));
  if (alu264) {
    data0_76800[(alu263+768)] = alu329;
  }
  if (alu264) {
    data0_76800[(alu263+769)] = alu330;
  }
  if (alu264) {
    data0_76800[(alu263+770)] = alu331;
  }
  if (alu264) {
    data0_76800[(alu263+771)] = alu332;
  }
  var alu345 = (acc1[4]+val68);
  var alu346 = (acc1[9]+val69);
  var alu347 = (acc1[14]+val70);
  var alu348 = (acc1[19]+val71);
  var alu349 = select(0.0f,alu345,(0.0f<alu345));
  var alu350 = select(0.0f,alu346,(0.0f<alu346));
  var alu351 = select(0.0f,alu347,(0.0f<alu347));
  var alu352 = select(0.0f,alu348,(0.0f<alu348));
  if (alu264) {
    data0_76800[(alu263+1024)] = alu349;
  }
  if (alu264) {
    data0_76800[(alu263+1025)] = alu350;
  }
  if (alu264) {
    data0_76800[(alu263+1026)] = alu351;
  }
  if (alu264) {
    data0_76800[(alu263+1027)] = alu352;
  }
}`;

const r_100_8_16_8_3_2_32 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32,768>;
@group(0) @binding(1)var<storage,read_write>data0_76800:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_76800:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_65536:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_256:array<f32>;
@compute @workgroup_size(16,8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,6>;
  var acc1: array<f32,6>;
  var gidx0 = i32(gindex.x); /* 8 */
  var gidx1 = i32(gindex.y); /* 100 */
  var lidx0 = i32(lindex.x); /* 16 */
  var lidx1 = i32(lindex.y); /* 8 */
  var cast0 = bitcast<u32>(gidx0);
  var alu0 = (gidx1*768);
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  acc0[4] = 0.0f;
  acc0[5] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 32; Ridx0++) {
    var alu7 = (lidx1+bitcast<i32>((bitcast<u32>(Ridx0)<<3u)));
    var alu8 = (alu7+alu0);
    var val0 = data1_76800[alu8];
    var alu9 = (alu7+bitcast<i32>((cast0<<13u))+bitcast<i32>((bitcast<u32>(lidx0)<<8u)));
    var val1 = data2_65536[alu9];
    var val2 = data2_65536[(alu9+4096)];
    var val3 = data1_76800[(alu8+256)];
    var val4 = data1_76800[(alu8+512)];
    acc0[0] = (acc0[0]+(val0*val1));
    acc0[1] = (acc0[1]+(val0*val2));
    acc0[2] = (acc0[2]+(val3*val1));
    acc0[3] = (acc0[3]+(val3*val2));
    acc0[4] = (acc0[4]+(val4*val1));
    acc0[5] = (acc0[5]+(val4*val2));
  }
  var alu17 = (lidx0*48);
  var alu18 = (alu17+(lidx1*6));
  temp0[(alu18+1)] = acc0[1];
  temp0[(alu18+2)] = acc0[2];
  temp0[(alu18+3)] = acc0[3];
  temp0[(alu18+4)] = acc0[4];
  temp0[(alu18+5)] = acc0[5];
  temp0[alu18] = acc0[0];
  workgroupBarrier();
  acc1[0] = 0.0f;
  acc1[1] = 0.0f;
  acc1[2] = 0.0f;
  acc1[3] = 0.0f;
  acc1[4] = 0.0f;
  acc1[5] = 0.0f;
  for (var Ridx105 = 0; Ridx105 < 8; Ridx105++) {
    var alu32 = (alu17+(Ridx105*6));
    var val5 = temp0[alu32];
    var val6 = temp0[(alu32+1)];
    var val7 = temp0[(alu32+2)];
    var val8 = temp0[(alu32+3)];
    var val9 = temp0[(alu32+4)];
    var val10 = temp0[(alu32+5)];
    acc1[0] = (acc1[0]+val5);
    acc1[1] = (acc1[1]+val6);
    acc1[2] = (acc1[2]+val7);
    acc1[3] = (acc1[3]+val8);
    acc1[4] = (acc1[4]+val9);
    acc1[5] = (acc1[5]+val10);
  }
  var alu40 = (lidx0+bitcast<i32>((cast0<<5u)));
  var val11 = data3_256[alu40];
  var val12 = data3_256[(alu40+16)];
  var alu41 = (alu40+alu0);
  var alu42 = (lidx1==0);
  if (alu42) {
    data0_76800[alu41] = (acc1[0]+val11);
  }
  if (alu42) {
    data0_76800[(alu41+16)] = (acc1[1]+val12);
  }
  if (alu42) {
    data0_76800[(alu41+256)] = (acc1[2]+val11);
  }
  if (alu42) {
    data0_76800[(alu41+272)] = (acc1[3]+val12);
  }
  if (alu42) {
    data0_76800[(alu41+512)] = (acc1[4]+val11);
  }
  if (alu42) {
    data0_76800[(alu41+528)] = (acc1[5]+val12);
  }
}`;

const r_30_8_8_4_5_2_256 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_76800:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_998400:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_76800:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_196608:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_768:array<f32>;
@compute @workgroup_size(8,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,10>;
  var gidx0 = i32(gindex.x); /* 8 */
  var gidx1 = i32(gindex.y); /* 30 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 4 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(lidx1);
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
    var val0 = data1_998400[alu11];
    var val1 = data2_76800[alu11];
    var val2 = data3_196608[(bitcast<i32>((cast0<<13u))+bitcast<i32>((cast1<<11u))+bitcast<i32>((bitcast<u32>(lidx0)<<8u))+Ridx0)];
    var alu12 = (alu11+256);
    var val3 = data1_998400[alu12];
    var alu13 = (alu11+1280);
    var val4 = data1_998400[alu13];
    var val5 = data2_76800[alu12];
    var val6 = data2_76800[alu13];
    var alu14 = (alu11+1536);
    var val7 = data1_998400[alu14];
    var alu15 = (alu11+512);
    var val8 = data2_76800[alu15];
    var val9 = data2_76800[alu14];
    var val10 = data1_998400[alu15];
    var alu16 = (alu11+1792);
    var val11 = data1_998400[alu16];
    var alu17 = (alu11+768);
    var val12 = data2_76800[alu17];
    var val13 = data2_76800[alu16];
    var val14 = data1_998400[alu17];
    var alu18 = (alu11+1024);
    var val15 = data1_998400[alu18];
    var alu19 = (alu11+2048);
    var val16 = data1_998400[alu19];
    var val17 = data2_76800[alu18];
    var val18 = data2_76800[alu19];
    var alu20 = (alu11+2304);
    var val19 = data1_998400[alu20];
    var val20 = data2_76800[alu20];
    acc0[0] = (acc0[0]+((val0+val1)*val2));
    acc0[1] = (acc0[1]+((val4+val6)*val2));
    acc0[2] = (acc0[2]+((val3+val5)*val2));
    acc0[3] = (acc0[3]+((val7+val9)*val2));
    acc0[4] = (acc0[4]+((val10+val8)*val2));
    acc0[5] = (acc0[5]+((val11+val13)*val2));
    acc0[6] = (acc0[6]+((val14+val12)*val2));
    acc0[7] = (acc0[7]+((val16+val18)*val2));
    acc0[8] = (acc0[8]+((val15+val17)*val2));
    acc0[9] = (acc0[9]+((val19+val20)*val2));
  }
  var alu32 = (lidx0+bitcast<i32>((cast0<<5u))+bitcast<i32>((cast1<<3u)));
  var val21 = data4_768[alu32];
  var alu33 = (alu32+alu0);
  data0_76800[alu33] = (acc0[0]+val21);
  data0_76800[(alu33+256)] = (acc0[2]+val21);
  data0_76800[(alu33+512)] = (acc0[4]+val21);
  data0_76800[(alu33+768)] = (acc0[6]+val21);
  data0_76800[(alu33+1024)] = (acc0[8]+val21);
  data0_76800[(alu33+1280)] = (acc0[1]+val21);
  data0_76800[(alu33+1536)] = (acc0[3]+val21);
  data0_76800[(alu33+1792)] = (acc0[5]+val21);
  data0_76800[(alu33+2048)] = (acc0[7]+val21);
  data0_76800[(alu33+2304)] = (acc0[9]+val21);
}`;

const r_75_16_8_4_2_4_64 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32,256>;
@group(0) @binding(1)var<storage,read_write>data0_76800:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_998400:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_76800:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_196608:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_768:array<f32>;
@compute @workgroup_size(8,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,8>;
  var acc1: array<f32,8>;
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 75 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 4 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<i32>((bitcast<u32>(gidx1)<<10u));
  var cast2 = bitcast<u32>(lidx0);
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  acc0[4] = 0.0f;
  acc0[5] = 0.0f;
  acc0[6] = 0.0f;
  acc0[7] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 64; Ridx0++) {
    var alu8 = (lidx1+bitcast<i32>((bitcast<u32>(Ridx0)<<2u)));
    var alu9 = (alu8+cast1);
    var val0 = data1_998400[alu9];
    var val1 = data2_76800[alu9];
    var alu10 = (alu8+bitcast<i32>((cast0<<12u))+bitcast<i32>((cast2<<9u)));
    var val2 = data3_196608[(alu10+65536)];
    var alu11 = (alu9+256);
    var val3 = data1_998400[alu11];
    var val4 = data2_76800[alu11];
    var alu12 = (alu9+512);
    var val5 = data1_998400[alu12];
    var val6 = data2_76800[alu12];
    var alu13 = (alu9+768);
    var val7 = data1_998400[alu13];
    var val8 = data2_76800[alu13];
    var val9 = data3_196608[(alu10+65792)];
    var alu14 = (val0+val1);
    var alu15 = (val3+val4);
    var alu16 = (val5+val6);
    var alu17 = (val7+val8);
    acc0[0] = (acc0[0]+(alu14*val2));
    acc0[1] = (acc0[1]+(alu15*val2));
    acc0[2] = (acc0[2]+(alu16*val2));
    acc0[3] = (acc0[3]+(alu17*val2));
    acc0[4] = (acc0[4]+(alu14*val9));
    acc0[5] = (acc0[5]+(alu15*val9));
    acc0[6] = (acc0[6]+(alu16*val9));
    acc0[7] = (acc0[7]+(alu17*val9));
  }
  var cast3 = bitcast<i32>((cast2<<5u));
  var alu27 = (cast3+bitcast<i32>((bitcast<u32>(lidx1)<<3u)));
  temp0[alu27] = acc0[0];
  temp0[(alu27+1)] = acc0[1];
  temp0[(alu27+2)] = acc0[2];
  temp0[(alu27+3)] = acc0[3];
  temp0[(alu27+4)] = acc0[4];
  temp0[(alu27+5)] = acc0[5];
  temp0[(alu27+6)] = acc0[6];
  temp0[(alu27+7)] = acc0[7];
  workgroupBarrier();
  acc1[0] = 0.0f;
  acc1[1] = 0.0f;
  acc1[2] = 0.0f;
  acc1[3] = 0.0f;
  acc1[4] = 0.0f;
  acc1[5] = 0.0f;
  acc1[6] = 0.0f;
  acc1[7] = 0.0f;
  for (var Ridx104 = 0; Ridx104 < 4; Ridx104++) {
    var alu45 = (cast3+bitcast<i32>((bitcast<u32>(Ridx104)<<3u)));
    var val10 = temp0[alu45];
    var val11 = temp0[(alu45+1)];
    var val12 = temp0[(alu45+2)];
    var val13 = temp0[(alu45+3)];
    var val14 = temp0[(alu45+4)];
    var val15 = temp0[(alu45+5)];
    var val16 = temp0[(alu45+6)];
    var val17 = temp0[(alu45+7)];
    acc1[0] = (acc1[0]+val10);
    acc1[1] = (acc1[1]+val11);
    acc1[2] = (acc1[2]+val12);
    acc1[3] = (acc1[3]+val13);
    acc1[4] = (acc1[4]+val14);
    acc1[5] = (acc1[5]+val15);
    acc1[6] = (acc1[6]+val16);
    acc1[7] = (acc1[7]+val17);
  }
  var alu55 = (bitcast<i32>((cast0<<4u))+bitcast<i32>((cast2<<1u)));
  var val18 = data4_768[(alu55+256)];
  var val19 = data4_768[(alu55+257)];
  var alu56 = (alu55+cast1);
  var alu57 = (lidx1==0);
  if (alu57) {
    data0_76800[alu56] = (acc1[0]+val18);
  }
  if (alu57) {
    data0_76800[(alu56+1)] = (acc1[4]+val19);
  }
  if (alu57) {
    data0_76800[(alu56+256)] = (acc1[1]+val18);
  }
  if (alu57) {
    data0_76800[(alu56+257)] = (acc1[5]+val19);
  }
  if (alu57) {
    data0_76800[(alu56+512)] = (acc1[2]+val18);
  }
  if (alu57) {
    data0_76800[(alu56+513)] = (acc1[6]+val19);
  }
  if (alu57) {
    data0_76800[(alu56+768)] = (acc1[3]+val18);
  }
  if (alu57) {
    data0_76800[(alu56+769)] = (acc1[7]+val19);
  }
}`;

const r_300_100_8_4_3_8 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32,96>;
@group(0) @binding(1)var<storage,read_write>data0_720000:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_76800:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_76800:array<f32>;
@compute @workgroup_size(8,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,3>;
  var acc1: array<f32,3>;
  var gidx0 = i32(gindex.x); /* 100 */
  var gidx1 = i32(gindex.y); /* 300 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 4 */
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 8; Ridx0++) {
    var alu3 = (lidx1+bitcast<i32>((bitcast<u32>(Ridx0)<<2u))+bitcast<i32>((bitcast<u32>(lidx0)<<5u)));
    var val0 = data1_76800[(alu3+bitcast<i32>((bitcast<u32>(gidx1)<<8u)))];
    var alu4 = (alu3+(gidx0*768));
    var val1 = data2_76800[alu4];
    var val2 = data2_76800[(alu4+256)];
    var val3 = data2_76800[(alu4+512)];
    acc0[0] = (acc0[0]+(val0*val1));
    acc0[1] = (acc0[1]+(val0*val2));
    acc0[2] = (acc0[2]+(val0*val3));
  }
  var alu9 = (lidx0*12);
  var alu10 = (alu9+(lidx1*3));
  temp0[(alu10+1)] = acc0[1];
  temp0[(alu10+2)] = acc0[2];
  temp0[alu10] = acc0[0];
  workgroupBarrier();
  acc1[0] = 0.0f;
  acc1[1] = 0.0f;
  acc1[2] = 0.0f;
  for (var Ridx106 = 0; Ridx106 < 4; Ridx106++) {
    var alu18 = (alu9+(Ridx106*3));
    var val4 = temp0[alu18];
    var val5 = temp0[(alu18+1)];
    var val6 = temp0[(alu18+2)];
    acc1[0] = (acc1[0]+val4);
    acc1[1] = (acc1[1]+val5);
    acc1[2] = (acc1[2]+val6);
  }
  var alu23 = ((gidx0*3)+(gidx1*300)+(lidx0*90000));
  var alu24 = (lidx1==0);
  if (alu24) {
    data0_720000[(alu23+1)] = (acc1[1]*0.17677669529663687f);
  }
  if (alu24) {
    data0_720000[(alu23+2)] = (acc1[2]*0.17677669529663687f);
  }
  if (alu24) {
    data0_720000[alu23] = (acc1[0]*0.17677669529663687f);
  }
}`;

const r_300_8_75_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_2400:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_720000:array<f32>;
@compute @workgroup_size(8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 300 */
  var lidx0 = i32(lindex.x); /* 8 */
  acc0[0] = (f32(-INFINITY));
  for (var Ridx0 = 0; Ridx0 < 75; Ridx0++) {
    var alu1 = ((gidx0*2400)+(lidx0*300)+bitcast<i32>((bitcast<u32>(Ridx0)<<2u)));
    var val0 = data1_720000[(alu1+1)];
    var val1 = data1_720000[(alu1+2)];
    var val2 = data1_720000[(alu1+3)];
    var val3 = data1_720000[alu1];
    var alu2 = select(acc0[0],val3,(acc0[0]<val3));
    var alu3 = select(alu2,val0,(alu2<val0));
    var alu4 = select(alu3,val1,(alu3<val1));
    var alu5 = select(alu4,val2,(alu4<val2));
    acc0[0] = alu5;
  }
  data0_2400[(lidx0+bitcast<i32>((bitcast<u32>(gidx0)<<3u)))] = acc0[0];
}`;

const r_150_8_2_75_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_2400:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_720000:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_2400:array<f32>;
@compute @workgroup_size(8,2) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 150 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 2 */
  var alu0 = (lidx0+bitcast<i32>((bitcast<u32>(gidx0)<<4u))+bitcast<i32>((bitcast<u32>(lidx1)<<3u)));
  var val0 = data2_2400[alu0];
  acc0[0] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 75; Ridx0++) {
    var alu2 = ((gidx0*4800)+(lidx1*2400)+(lidx0*300)+bitcast<i32>((bitcast<u32>(Ridx0)<<2u)));
    var val1 = data1_720000[(alu2+1)];
    var val2 = data1_720000[(alu2+2)];
    var val3 = data1_720000[(alu2+3)];
    var val4 = data1_720000[alu2];
    acc0[0] = (acc0[0]+exp2(((val4-val0)*1.4426950408889634f))+exp2(((val1-val0)*1.4426950408889634f))+exp2(((val2-val0)*1.4426950408889634f))+exp2(((val3-val0)*1.4426950408889634f)));
  }
  data0_2400[alu0] = acc0[0];
}`;

const r_2_8_100_16_3_300 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_76800:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_720000:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_2400:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_2400:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_76800:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,3>;
  var gidx0 = i32(gindex.x); /* 100 */
  var gidx1 = i32(gindex.y); /* 8 */
  var alu0 = ((gidx0*3)+(gidx1*300));
  var alu1 = (alu0+1);
  var val0 = data2_2400[alu1];
  var alu2 = (alu0+2);
  var val1 = data2_2400[alu2];
  var val2 = data2_2400[alu0];
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 16 */
  var alu3 = (lidx0+bitcast<i32>((bitcast<u32>(gidx2)<<4u))+bitcast<i32>((bitcast<u32>(gidx1)<<5u)));
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 300; Ridx0++) {
    var alu7 = ((gidx0*900)+Ridx0+(gidx1*90000));
    var val3 = data1_720000[alu7];
    var val4 = data4_76800[(alu3+bitcast<i32>((bitcast<u32>(Ridx0)<<8u)))];
    var val5 = data1_720000[(alu7+300)];
    var val6 = data1_720000[(alu7+600)];
    acc0[0] = (acc0[0]+(exp2(((val3-val2)*1.4426950408889634f))*val4));
    acc0[1] = (acc0[1]+(exp2(((val5-val0)*1.4426950408889634f))*val4));
    acc0[2] = (acc0[2]+(exp2(((val6-val1)*1.4426950408889634f))*val4));
  }
  var val7 = data3_2400[alu0];
  var val8 = data3_2400[alu1];
  var val9 = data3_2400[alu2];
  var alu12 = (alu3+(gidx0*768));
  data0_76800[alu12] = (acc0[0]*(1/val7));
  data0_76800[(alu12+256)] = (acc0[1]*(1/val8));
  data0_76800[(alu12+512)] = (acc0[2]*(1/val9));
}`;

const r_60_2_16_8_5_256 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_76800:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_998400:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_76800:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_65536:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_256:array<f32>;
@compute @workgroup_size(16,8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,5>;
  var gidx0 = i32(gindex.x); /* 2 */
  var gidx1 = i32(gindex.y); /* 60 */
  var lidx0 = i32(lindex.x); /* 16 */
  var lidx1 = i32(lindex.y); /* 8 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(lidx1);
  var alu0 = (gidx1*1280);
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  acc0[4] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 256; Ridx0++) {
    var alu6 = (alu0+Ridx0);
    var val0 = data2_76800[alu6];
    var val1 = data3_65536[(bitcast<i32>((cast0<<15u))+bitcast<i32>((cast1<<12u))+bitcast<i32>((bitcast<u32>(lidx0)<<8u))+Ridx0)];
    var val2 = data2_76800[(alu6+256)];
    var val3 = data2_76800[(alu6+512)];
    var val4 = data2_76800[(alu6+768)];
    var val5 = data2_76800[(alu6+1024)];
    acc0[0] = (acc0[0]+(val0*val1));
    acc0[1] = (acc0[1]+(val2*val1));
    acc0[2] = (acc0[2]+(val3*val1));
    acc0[3] = (acc0[3]+(val4*val1));
    acc0[4] = (acc0[4]+(val5*val1));
  }
  var alu13 = (lidx0+bitcast<i32>((cast0<<7u))+bitcast<i32>((cast1<<4u)));
  var alu14 = (alu13+alu0);
  var val6 = data1_998400[alu14];
  var val7 = data4_256[alu13];
  var alu15 = (alu14+256);
  var val8 = data1_998400[alu15];
  var alu16 = (alu14+512);
  var val9 = data1_998400[alu16];
  var alu17 = (alu14+768);
  var val10 = data1_998400[alu17];
  var alu18 = (alu14+1024);
  var val11 = data1_998400[alu18];
  data0_76800[alu14] = (val6+acc0[0]+val7);
  data0_76800[alu15] = (val8+acc0[1]+val7);
  data0_76800[alu16] = (val9+acc0[2]+val7);
  data0_76800[alu17] = (val10+acc0[3]+val7);
  data0_76800[alu18] = (val11+acc0[4]+val7);
}`;

const r_300_16_16 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32,16>;
@group(0) @binding(1)var<storage,read_write>data0_300:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_76800:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,1>;
  var acc1: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 300 */
  var lidx0 = i32(lindex.x); /* 16 */
  acc0[0] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 16; Ridx0++) {
    var val0 = data1_76800[(bitcast<i32>((bitcast<u32>(lidx0)<<4u))+Ridx0+bitcast<i32>((bitcast<u32>(gidx0)<<8u)))];
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
    data0_300[gidx0] = (acc1[0]*0.00390625f);
  }
}`;

const r_300_16_16n1 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
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

const E_300_32_8 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_76800:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_76800:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_300:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_300:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_256:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_256:array<f32>;
@compute @workgroup_size(8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 32 */
  var gidx1 = i32(gindex.y); /* 300 */
  var lidx0 = i32(lindex.x); /* 8 */
  var alu0 = (lidx0+bitcast<i32>((bitcast<u32>(gidx0)<<3u)));
  var alu1 = (alu0+bitcast<i32>((bitcast<u32>(gidx1)<<8u)));
  var val0 = data1_76800[alu1];
  var val1 = data2_300[gidx1];
  var val2 = data3_300[gidx1];
  var val3 = data4_256[alu0];
  var val4 = data5_256[alu0];
  data0_76800[alu1] = (((val0-val1)*val2*val3)+val4);
}`;

const r_300_2_2_16_16_16 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32,256>;
@group(0) @binding(1)var<storage,read_write>data0_19200:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_15600:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_1200:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_76800:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_76800:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_16384:array<f32>;
@group(0) @binding(7)var<storage,read_write>data6_64:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,16>;
  var acc1: array<f32,16>;
  var gidx0 = i32(gindex.x); /* 2 */
  var gidx1 = i32(gindex.y); /* 2 */
  var gidx2 = i32(gindex.z); /* 300 */
  var lidx0 = i32(lindex.x); /* 16 */
  var cast0 = bitcast<u32>(gidx1);
  var cast1 = bitcast<u32>(gidx2);
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
    var alu17 = (alu16+bitcast<i32>((cast1<<8u)));
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
  var cast2 = bitcast<i32>((bitcast<u32>(lidx0)<<4u));
  temp0[cast2] = acc0[0];
  temp0[(cast2+1)] = acc0[1];
  temp0[(cast2+2)] = acc0[2];
  temp0[(cast2+3)] = acc0[3];
  temp0[(cast2+4)] = acc0[4];
  temp0[(cast2+5)] = acc0[5];
  temp0[(cast2+6)] = acc0[6];
  temp0[(cast2+7)] = acc0[7];
  temp0[(cast2+8)] = acc0[8];
  temp0[(cast2+9)] = acc0[9];
  temp0[(cast2+10)] = acc0[10];
  temp0[(cast2+11)] = acc0[11];
  temp0[(cast2+12)] = acc0[12];
  temp0[(cast2+13)] = acc0[13];
  temp0[(cast2+14)] = acc0[14];
  temp0[(cast2+15)] = acc0[15];
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
    var cast3 = bitcast<i32>((bitcast<u32>(Ridx105)<<4u));
    var val18 = temp0[cast3];
    var val19 = temp0[(cast3+1)];
    var val20 = temp0[(cast3+2)];
    var val21 = temp0[(cast3+3)];
    var val22 = temp0[(cast3+4)];
    var val23 = temp0[(cast3+5)];
    var val24 = temp0[(cast3+6)];
    var val25 = temp0[(cast3+7)];
    var val26 = temp0[(cast3+8)];
    var val27 = temp0[(cast3+9)];
    var val28 = temp0[(cast3+10)];
    var val29 = temp0[(cast3+11)];
    var val30 = temp0[(cast3+12)];
    var val31 = temp0[(cast3+13)];
    var val32 = temp0[(cast3+14)];
    var val33 = temp0[(cast3+15)];
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
  var cast4 = bitcast<i32>((cast1<<2u));
  var alu87 = (gidx0+cast4);
  var val34 = data1_15600[alu87];
  var val35 = data2_1200[alu87];
  var alu88 = (alu87+2);
  var val36 = data2_1200[alu88];
  var alu89 = (gidx0+bitcast<i32>((cast0<<1u)));
  var val37 = data6_64[alu89];
  var val38 = data1_15600[alu88];
  var val39 = data6_64[(alu89+4)];
  var val40 = data6_64[(alu89+8)];
  var val41 = data6_64[(alu89+12)];
  var val42 = data6_64[(alu89+16)];
  var val43 = data6_64[(alu89+20)];
  var val44 = data6_64[(alu89+24)];
  var val45 = data6_64[(alu89+28)];
  var val46 = data6_64[(alu89+32)];
  var val47 = data6_64[(alu89+36)];
  var val48 = data6_64[(alu89+40)];
  var val49 = data6_64[(alu89+44)];
  var val50 = data6_64[(alu89+48)];
  var val51 = data6_64[(alu89+52)];
  var val52 = data6_64[(alu89+56)];
  var val53 = data6_64[(alu89+60)];
  var alu90 = (alu89+cast4);
  var alu91 = (lidx0==0);
  var alu92 = ((val34*val36)+val35);
  var alu93 = (exp2((val38*1.4426950408889634f))*val36);
  if (alu91) {
    data0_19200[alu90] = ((2.0f*(alu92+((acc1[0]+val37)*alu93*0.25f)))+-1.0f);
  }
  if (alu91) {
    data0_19200[(alu90+1200)] = ((2.0f*(alu92+((acc1[1]+val39)*alu93*0.25f)))+-1.0f);
  }
  if (alu91) {
    data0_19200[(alu90+2400)] = ((2.0f*(alu92+((acc1[2]+val40)*alu93*0.25f)))+-1.0f);
  }
  if (alu91) {
    data0_19200[(alu90+3600)] = ((2.0f*(alu92+((acc1[3]+val41)*alu93*0.25f)))+-1.0f);
  }
  if (alu91) {
    data0_19200[(alu90+4800)] = ((2.0f*(alu92+((acc1[4]+val42)*alu93*0.25f)))+-1.0f);
  }
  if (alu91) {
    data0_19200[(alu90+6000)] = ((2.0f*(alu92+((acc1[5]+val43)*alu93*0.25f)))+-1.0f);
  }
  if (alu91) {
    data0_19200[(alu90+7200)] = ((2.0f*(alu92+((acc1[6]+val44)*alu93*0.25f)))+-1.0f);
  }
  if (alu91) {
    data0_19200[(alu90+8400)] = ((2.0f*(alu92+((acc1[7]+val45)*alu93*0.25f)))+-1.0f);
  }
  if (alu91) {
    data0_19200[(alu90+9600)] = ((2.0f*(alu92+((acc1[8]+val46)*alu93*0.25f)))+-1.0f);
  }
  if (alu91) {
    data0_19200[(alu90+10800)] = ((2.0f*(alu92+((acc1[9]+val47)*alu93*0.25f)))+-1.0f);
  }
  if (alu91) {
    data0_19200[(alu90+12000)] = ((2.0f*(alu92+((acc1[10]+val48)*alu93*0.25f)))+-1.0f);
  }
  if (alu91) {
    data0_19200[(alu90+13200)] = ((2.0f*(alu92+((acc1[11]+val49)*alu93*0.25f)))+-1.0f);
  }
  if (alu91) {
    data0_19200[(alu90+14400)] = ((2.0f*(alu92+((acc1[12]+val50)*alu93*0.25f)))+-1.0f);
  }
  if (alu91) {
    data0_19200[(alu90+15600)] = ((2.0f*(alu92+((acc1[13]+val51)*alu93*0.25f)))+-1.0f);
  }
  if (alu91) {
    data0_19200[(alu90+16800)] = ((2.0f*(alu92+((acc1[14]+val52)*alu93*0.25f)))+-1.0f);
  }
  if (alu91) {
    data0_19200[(alu90+18000)] = ((2.0f*(alu92+((acc1[15]+val53)*alu93*0.25f)))+-1.0f);
  }
}`;

const r_300_2_16_256 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_9600:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_76800:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_76800:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_8192:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_32:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 2 */
  var gidx1 = i32(gindex.y); /* 300 */
  var lidx0 = i32(lindex.x); /* 16 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(gidx1);
  acc0[0] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 256; Ridx0++) {
    var alu1 = (bitcast<i32>((cast1<<8u))+Ridx0);
    var val0 = data1_76800[alu1];
    var val1 = data2_76800[alu1];
    var val2 = data3_8192[(bitcast<i32>((cast0<<12u))+bitcast<i32>((bitcast<u32>(lidx0)<<8u))+Ridx0)];
    acc0[0] = (acc0[0]+((val0+val1)*val2));
  }
  var alu4 = (lidx0+bitcast<i32>((cast0<<4u)));
  var val3 = data4_32[alu4];
  data0_9600[(alu4+bitcast<i32>((cast1<<5u)))] = (acc0[0]+val3);
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

const r_300_8_16_2_2 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
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
  var gidx0 = i32(gindex.x); /* 8 */
  var gidx1 = i32(gindex.y); /* 300 */
  var lidx1 = i32(lindex.y); /* 2 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(gidx1);
  var alu0 = (lidx1+bitcast<i32>((cast0<<1u))+bitcast<i32>((cast1<<4u)));
  var val0 = data4_4800[alu0];
  var lidx0 = i32(lindex.x); /* 16 */
  var cast2 = bitcast<u32>(lidx1);
  var alu1 = ((gidx0*18432)+(lidx1*9216)+(lidx0*576));
  acc0[0] = 0.0f;
  for (var Ridx4 = 0; Ridx4 < 2; Ridx4++) {
    var alu3 = ((gidx0*2400)+(lidx1*1200)+bitcast<i32>((cast1<<2u))+bitcast<i32>((bitcast<u32>(Ridx4)<<1u)));
    var val1 = data1_19200[(alu3+1)];
    var val2 = data1_19200[alu3];
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
    var alu20 = ((i32((alu7*2.3283064365386963e-10f)))-(i32(((alu7<0.0f)&(cast3!=0)))));
    var alu21 = ((i32((alu12*2.3283064365386963e-10f)))-(i32(((alu12<0.0f)&(cast5!=0)))));
    var alu22 = select(alu20,alu21,alu14);
    var alu23 = ((alu22<-1)|((alu22==-1)&(cast7<4294967295u)));
    var alu24 = select(alu16,0,alu23);
    var alu25 = select((alu22+(i32((bitcast<u32>(alu16)<cast7)))),0,alu23);
    var alu26 = ((0<alu25)|((0==alu25)&(23u<bitcast<u32>(alu24))));
    var alu27 = select(alu24,23,alu26);
    var cast9 = bitcast<u32>(alu27);
    var alu28 = ((cast9>>16u)*24u);
    var cast10 = bitcast<i32>((alu28<<16u));
    var alu29 = ((i32((alu11*2.3283064365386963e-10f)))-(i32(((alu11<0.0f)&(cast4!=0)))));
    var alu30 = ((i32((alu13*2.3283064365386963e-10f)))-(i32(((alu13<0.0f)&(cast6!=0)))));
    var alu31 = select(alu29,alu30,alu17);
    var alu32 = ((alu31<-1)|((alu31==-1)&(cast8<4294967295u)));
    var alu33 = select(alu19,0,alu32);
    var alu34 = (cast10+bitcast<i32>(((cast9&65535u)*24u)));
    var alu35 = select((alu31+(i32((bitcast<u32>(alu19)<cast8)))),0,alu32);
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
    var alu44 = (alu42|(alu43&(cast8<0u)));
    var alu45 = select(alu18,0,alu44);
    var alu46 = select(alu31,0,alu44);
    var alu47 = ((0<alu46)|((0==alu46)&(23u<bitcast<u32>(alu45))));
    var alu48 = select(alu45,23,alu47);
    var cast14 = bitcast<u32>((alu34+alu48));
    var alu49 = select(alu46,0,alu47);
    var alu50 = (alu39+alu49+(i32((cast14<cast13))));
    var val4 = select(0.0f, data2_147456[((i32(cast14))+alu1)], (((-1<alu50)|((-1==alu50)&(4294967295u<cast14)))&((alu50<0)|((alu50==0)&(cast14<576u)))));
    var alu51 = (alu22<0);
    var alu52 = (alu22==0);
    var alu53 = (alu51|(alu52&(cast7<0u)));
    var alu54 = select(alu15,0,alu53);
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
    var val7 = data3_9600[(bitcast<i32>((cast0<<2u))+bitcast<i32>((cast2<<1u))+Ridx4+bitcast<i32>((cast1<<5u)))];
    var alu64 = (-1<alu22);
    var alu65 = (-1<alu31);
    var alu66 = (-1==alu22);
    var alu67 = (-1==alu31);
    var alu68 = (alu51|(alu52&(cast7<23u)));
    var alu69 = (alu64|(alu66&(4294967294u<cast7)));
    var alu70 = ((alu65|(alu67&(4294967294u<cast8)))&(alu42|(alu43&(cast8<23u))));
    var alu71 = (alu51|(alu52&(cast7<24u)));
    var alu72 = (alu64|(alu66&(4294967295u<cast7)));
    var alu73 = ((alu65|(alu67&(4294967295u<cast8)))&(alu42|(alu43&(cast8<24u))));
    var alu74 = select((((f32(alu20))*4294967296.0f)+(f32(bitcast<u32>(cast3)))),(f32(cast3)),(((alu20==0)&(-1<cast3))|((alu20==-1)&(cast3<0))));
    var alu75 = select((((f32(alu21))*4294967296.0f)+(f32(bitcast<u32>(cast5)))),(f32(cast5)),(((alu21==0)&(-1<cast5))|((alu21==-1)&(cast5<0))));
    var alu76 = select(alu74,alu75,alu14);
    var alu77 = ((alu4*-12.0f)+alu76+1.5f);
    var alu78 = select((((f32(alu29))*4294967296.0f)+(f32(bitcast<u32>(cast4)))),(f32(cast4)),(((alu29==0)&(-1<cast4))|((alu29==-1)&(cast4<0))));
    var alu79 = select((((f32(alu30))*4294967296.0f)+(f32(bitcast<u32>(cast6)))),(f32(cast6)),(((alu30==0)&(-1<cast6))|((alu30==-1)&(cast6<0))));
    var alu80 = select(alu78,alu79,alu17);
    var alu81 = ((alu8*-12.0f)+alu80+1.5f);
    var alu82 = ((alu5-alu76)+-0.5f);
    var alu83 = ((alu9-alu80)+-0.5f);
    acc0[0] = (acc0[0]+(((val6*alu81*alu77*(f32((alu73&alu72&alu71))))+(val4*alu81*alu82*(f32((alu73&alu69&alu68))))+(val5*alu83*alu77*(f32((alu70&alu72&alu71))))+(val3*alu83*alu82*(f32((alu70&alu69&alu68)))))*exp2(((val7-val0)*1.4426950408889634f))));
  }
  var val8 = data5_4800[alu0];
  data0_76800[(lidx0+bitcast<i32>((cast0<<5u))+bitcast<i32>((cast2<<4u))+bitcast<i32>((cast1<<8u)))] = (acc0[0]*(1/val8));
}`;

const r_100_16_16_3_256n1 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_76800:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_76800:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_76800:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_65536:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_256:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,3>;
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 100 */
  var lidx0 = i32(lindex.x); /* 16 */
  var cast0 = bitcast<u32>(gidx0);
  var alu0 = (gidx1*768);
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 256; Ridx0++) {
    var alu4 = (alu0+Ridx0);
    var val0 = data2_76800[alu4];
    var val1 = data3_65536[(bitcast<i32>((cast0<<12u))+bitcast<i32>((bitcast<u32>(lidx0)<<8u))+Ridx0)];
    var val2 = data2_76800[(alu4+256)];
    var val3 = data2_76800[(alu4+512)];
    acc0[0] = (acc0[0]+(val0*val1));
    acc0[1] = (acc0[1]+(val2*val1));
    acc0[2] = (acc0[2]+(val3*val1));
  }
  var alu9 = (lidx0+bitcast<i32>((cast0<<4u)));
  var alu10 = (alu9+alu0);
  var val4 = data1_76800[alu10];
  var val5 = data4_256[alu9];
  var alu11 = (alu10+256);
  var val6 = data1_76800[alu11];
  var alu12 = (alu10+512);
  var val7 = data1_76800[alu12];
  data0_76800[alu10] = (val4+acc0[0]+val5);
  data0_76800[alu11] = (val6+acc0[1]+val5);
  data0_76800[alu12] = (val7+acc0[2]+val5);
}`;

const r_30_64_16_5_2_2_256 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_614400:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_76800:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_524288:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_2048:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,20>;
  var gidx0 = i32(gindex.x); /* 64 */
  var gidx1 = i32(gindex.y); /* 30 */
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
    var alu20 = ((gidx1*2560)+Ridx0);
    var val0 = data1_76800[(alu20+1024)];
    var val1 = data1_76800[alu20];
    var alu21 = (bitcast<i32>((cast0<<13u))+bitcast<i32>((bitcast<u32>(lidx0)<<8u))+Ridx0);
    var val2 = data2_524288[alu21];
    var val3 = data1_76800[(alu20+1280)];
    var val4 = data2_524288[(alu21+4096)];
    var val5 = data1_76800[(alu20+256)];
    var val6 = data1_76800[(alu20+1536)];
    var val7 = data1_76800[(alu20+512)];
    var val8 = data1_76800[(alu20+1792)];
    var val9 = data1_76800[(alu20+768)];
    var val10 = data1_76800[(alu20+2048)];
    var val11 = data1_76800[(alu20+2304)];
    acc0[0] = (acc0[0]+(val1*val2));
    acc0[1] = (acc0[1]+(val3*val2));
    acc0[2] = (acc0[2]+(val1*val4));
    acc0[3] = (acc0[3]+(val3*val4));
    acc0[4] = (acc0[4]+(val5*val2));
    acc0[5] = (acc0[5]+(val6*val2));
    acc0[6] = (acc0[6]+(val5*val4));
    acc0[7] = (acc0[7]+(val6*val4));
    acc0[8] = (acc0[8]+(val7*val2));
    acc0[9] = (acc0[9]+(val8*val2));
    acc0[10] = (acc0[10]+(val7*val4));
    acc0[11] = (acc0[11]+(val8*val4));
    acc0[12] = (acc0[12]+(val9*val2));
    acc0[13] = (acc0[13]+(val10*val2));
    acc0[14] = (acc0[14]+(val9*val4));
    acc0[15] = (acc0[15]+(val10*val4));
    acc0[16] = (acc0[16]+(val0*val2));
    acc0[17] = (acc0[17]+(val11*val2));
    acc0[18] = (acc0[18]+(val0*val4));
    acc0[19] = (acc0[19]+(val11*val4));
  }
  var alu43 = (lidx0+bitcast<i32>((cast0<<5u)));
  var val12 = data3_2048[alu43];
  var val13 = data3_2048[(alu43+16)];
  var alu44 = (alu43+(gidx1*20480));
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
  data0_614400[(alu44+10240)] = alu66;
  data0_614400[(alu44+10256)] = alu68;
  data0_614400[(alu44+12288)] = alu70;
  data0_614400[(alu44+12304)] = alu72;
  data0_614400[(alu44+14336)] = alu74;
  data0_614400[(alu44+14352)] = alu76;
  data0_614400[(alu44+16384)] = alu78;
  data0_614400[(alu44+16400)] = alu80;
  data0_614400[(alu44+18432)] = alu82;
  data0_614400[(alu44+18448)] = alu84;
}`;

const r_20_4_16_2_5_3_2_2048 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_76800:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_76800:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_614400:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_524288:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_256:array<f32>;
@compute @workgroup_size(16,2) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,30>;
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 20 */
  var lidx0 = i32(lindex.x); /* 16 */
  var lidx1 = i32(lindex.y); /* 2 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(lidx1);
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
  for (var Ridx0 = 0; Ridx0 < 2048; Ridx0++) {
    var alu30 = ((gidx1*30720)+Ridx0);
    var val0 = data2_614400[alu30];
    var alu31 = (bitcast<i32>((cast0<<17u))+bitcast<i32>((cast1<<15u))+bitcast<i32>((bitcast<u32>(lidx0)<<11u))+Ridx0);
    var val1 = data3_524288[alu31];
    var val2 = data3_524288[(alu31+65536)];
    var val3 = data2_614400[(alu30+10240)];
    var val4 = data2_614400[(alu30+2048)];
    var val5 = data2_614400[(alu30+22528)];
    var val6 = data2_614400[(alu30+4096)];
    var val7 = data2_614400[(alu30+14336)];
    var val8 = data2_614400[(alu30+20480)];
    var val9 = data2_614400[(alu30+12288)];
    var val10 = data2_614400[(alu30+24576)];
    var val11 = data2_614400[(alu30+6144)];
    var val12 = data2_614400[(alu30+26624)];
    var val13 = data2_614400[(alu30+8192)];
    var val14 = data2_614400[(alu30+16384)];
    var val15 = data2_614400[(alu30+18432)];
    var val16 = data2_614400[(alu30+28672)];
    acc0[0] = (acc0[0]+(val0*val1));
    acc0[1] = (acc0[1]+(val0*val2));
    acc0[2] = (acc0[2]+(val3*val1));
    acc0[3] = (acc0[3]+(val3*val2));
    acc0[4] = (acc0[4]+(val8*val1));
    acc0[5] = (acc0[5]+(val8*val2));
    acc0[6] = (acc0[6]+(val4*val1));
    acc0[7] = (acc0[7]+(val4*val2));
    acc0[8] = (acc0[8]+(val9*val1));
    acc0[9] = (acc0[9]+(val9*val2));
    acc0[10] = (acc0[10]+(val5*val1));
    acc0[11] = (acc0[11]+(val5*val2));
    acc0[12] = (acc0[12]+(val6*val1));
    acc0[13] = (acc0[13]+(val6*val2));
    acc0[14] = (acc0[14]+(val7*val1));
    acc0[15] = (acc0[15]+(val7*val2));
    acc0[16] = (acc0[16]+(val10*val1));
    acc0[17] = (acc0[17]+(val10*val2));
    acc0[18] = (acc0[18]+(val11*val1));
    acc0[19] = (acc0[19]+(val11*val2));
    acc0[20] = (acc0[20]+(val14*val1));
    acc0[21] = (acc0[21]+(val14*val2));
    acc0[22] = (acc0[22]+(val12*val1));
    acc0[23] = (acc0[23]+(val12*val2));
    acc0[24] = (acc0[24]+(val13*val1));
    acc0[25] = (acc0[25]+(val13*val2));
    acc0[26] = (acc0[26]+(val15*val1));
    acc0[27] = (acc0[27]+(val15*val2));
    acc0[28] = (acc0[28]+(val16*val1));
    acc0[29] = (acc0[29]+(val16*val2));
  }
  var alu63 = (lidx0+bitcast<i32>((cast0<<6u))+bitcast<i32>((cast1<<4u)));
  var alu64 = (alu63+(gidx1*3840));
  var val17 = data1_76800[alu64];
  var val18 = data4_256[alu63];
  var alu65 = (alu64+32);
  var val19 = data1_76800[alu65];
  var val20 = data4_256[(alu63+32)];
  var alu66 = (alu64+256);
  var val21 = data1_76800[alu66];
  var alu67 = (alu64+288);
  var val22 = data1_76800[alu67];
  var alu68 = (alu64+512);
  var val23 = data1_76800[alu68];
  var alu69 = (alu64+544);
  var val24 = data1_76800[alu69];
  var alu70 = (alu64+768);
  var val25 = data1_76800[alu70];
  var alu71 = (alu64+800);
  var val26 = data1_76800[alu71];
  var alu72 = (alu64+1024);
  var val27 = data1_76800[alu72];
  var alu73 = (alu64+1056);
  var val28 = data1_76800[alu73];
  var alu74 = (alu64+1280);
  var val29 = data1_76800[alu74];
  var alu75 = (alu64+1312);
  var val30 = data1_76800[alu75];
  var alu76 = (alu64+1536);
  var val31 = data1_76800[alu76];
  var alu77 = (alu64+1568);
  var val32 = data1_76800[alu77];
  var alu78 = (alu64+1792);
  var val33 = data1_76800[alu78];
  var alu79 = (alu64+1824);
  var val34 = data1_76800[alu79];
  var alu80 = (alu64+2048);
  var val35 = data1_76800[alu80];
  var alu81 = (alu64+2080);
  var val36 = data1_76800[alu81];
  var alu82 = (alu64+2304);
  var val37 = data1_76800[alu82];
  var alu83 = (alu64+2336);
  var val38 = data1_76800[alu83];
  var alu84 = (alu64+2560);
  var val39 = data1_76800[alu84];
  var alu85 = (alu64+2592);
  var val40 = data1_76800[alu85];
  var alu86 = (alu64+2816);
  var val41 = data1_76800[alu86];
  var alu87 = (alu64+2848);
  var val42 = data1_76800[alu87];
  var alu88 = (alu64+3072);
  var val43 = data1_76800[alu88];
  var alu89 = (alu64+3104);
  var val44 = data1_76800[alu89];
  var alu90 = (alu64+3328);
  var val45 = data1_76800[alu90];
  var alu91 = (alu64+3360);
  var val46 = data1_76800[alu91];
  var alu92 = (alu64+3584);
  var val47 = data1_76800[alu92];
  var alu93 = (alu64+3616);
  var val48 = data1_76800[alu93];
  data0_76800[alu64] = (val17+acc0[0]+val18);
  data0_76800[alu65] = (val19+acc0[1]+val20);
  data0_76800[alu66] = (val21+acc0[6]+val18);
  data0_76800[alu67] = (val22+acc0[7]+val20);
  data0_76800[alu68] = (val23+acc0[12]+val18);
  data0_76800[alu69] = (val24+acc0[13]+val20);
  data0_76800[alu70] = (val25+acc0[18]+val18);
  data0_76800[alu71] = (val26+acc0[19]+val20);
  data0_76800[alu72] = (val27+acc0[24]+val18);
  data0_76800[alu73] = (val28+acc0[25]+val20);
  data0_76800[alu74] = (val29+acc0[2]+val18);
  data0_76800[alu75] = (val30+acc0[3]+val20);
  data0_76800[alu76] = (val31+acc0[8]+val18);
  data0_76800[alu77] = (val32+acc0[9]+val20);
  data0_76800[alu78] = (val33+acc0[14]+val18);
  data0_76800[alu79] = (val34+acc0[15]+val20);
  data0_76800[alu80] = (val35+acc0[20]+val18);
  data0_76800[alu81] = (val36+acc0[21]+val20);
  data0_76800[alu82] = (val37+acc0[26]+val18);
  data0_76800[alu83] = (val38+acc0[27]+val20);
  data0_76800[alu84] = (val39+acc0[4]+val18);
  data0_76800[alu85] = (val40+acc0[5]+val20);
  data0_76800[alu86] = (val41+acc0[10]+val18);
  data0_76800[alu87] = (val42+acc0[11]+val20);
  data0_76800[alu88] = (val43+acc0[16]+val18);
  data0_76800[alu89] = (val44+acc0[17]+val20);
  data0_76800[alu90] = (val45+acc0[22]+val18);
  data0_76800[alu91] = (val46+acc0[23]+val20);
  data0_76800[alu92] = (val47+acc0[28]+val18);
  data0_76800[alu93] = (val48+acc0[29]+val20);
}`;

const r_100_16_3_16_256 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_76800:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_76800:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_76800:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_196608:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_768:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,48>;
  var gidx0 = i32(gindex.x); /* 100 */
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
    var val0 = data1_76800[alu49];
    var val1 = data2_76800[alu49];
    var alu50 = (bitcast<i32>((bitcast<u32>(lidx0)<<8u))+Ridx0);
    var val2 = data3_196608[alu50];
    var val3 = data3_196608[(alu50+4096)];
    var val4 = data3_196608[(alu50+8192)];
    var val5 = data3_196608[(alu50+12288)];
    var val6 = data3_196608[(alu50+16384)];
    var val7 = data3_196608[(alu50+20480)];
    var val8 = data3_196608[(alu50+24576)];
    var val9 = data3_196608[(alu50+28672)];
    var val10 = data3_196608[(alu50+32768)];
    var val11 = data3_196608[(alu50+36864)];
    var val12 = data3_196608[(alu50+40960)];
    var val13 = data3_196608[(alu50+45056)];
    var val14 = data3_196608[(alu50+49152)];
    var val15 = data3_196608[(alu50+53248)];
    var val16 = data3_196608[(alu50+57344)];
    var val17 = data3_196608[(alu50+61440)];
    var alu51 = (alu49+256);
    var val18 = data1_76800[alu51];
    var val19 = data2_76800[alu51];
    var alu52 = (alu49+512);
    var val20 = data1_76800[alu52];
    var val21 = data2_76800[alu52];
    var alu53 = (val0+val1);
    var alu54 = (val18+val19);
    var alu55 = (val20+val21);
    acc0[0] = (acc0[0]+(alu53*val2));
    acc0[1] = (acc0[1]+(alu53*val3));
    acc0[2] = (acc0[2]+(alu53*val4));
    acc0[3] = (acc0[3]+(alu53*val5));
    acc0[4] = (acc0[4]+(alu53*val6));
    acc0[5] = (acc0[5]+(alu53*val7));
    acc0[6] = (acc0[6]+(alu53*val8));
    acc0[7] = (acc0[7]+(alu53*val9));
    acc0[8] = (acc0[8]+(alu53*val10));
    acc0[9] = (acc0[9]+(alu53*val11));
    acc0[10] = (acc0[10]+(alu53*val12));
    acc0[11] = (acc0[11]+(alu53*val13));
    acc0[12] = (acc0[12]+(alu53*val14));
    acc0[13] = (acc0[13]+(alu53*val15));
    acc0[14] = (acc0[14]+(alu53*val16));
    acc0[15] = (acc0[15]+(alu53*val17));
    acc0[16] = (acc0[16]+(alu54*val2));
    acc0[17] = (acc0[17]+(alu54*val3));
    acc0[18] = (acc0[18]+(alu54*val4));
    acc0[19] = (acc0[19]+(alu54*val5));
    acc0[20] = (acc0[20]+(alu54*val6));
    acc0[21] = (acc0[21]+(alu54*val7));
    acc0[22] = (acc0[22]+(alu54*val8));
    acc0[23] = (acc0[23]+(alu54*val9));
    acc0[24] = (acc0[24]+(alu54*val10));
    acc0[25] = (acc0[25]+(alu54*val11));
    acc0[26] = (acc0[26]+(alu54*val12));
    acc0[27] = (acc0[27]+(alu54*val13));
    acc0[28] = (acc0[28]+(alu54*val14));
    acc0[29] = (acc0[29]+(alu54*val15));
    acc0[30] = (acc0[30]+(alu54*val16));
    acc0[31] = (acc0[31]+(alu54*val17));
    acc0[32] = (acc0[32]+(alu55*val2));
    acc0[33] = (acc0[33]+(alu55*val3));
    acc0[34] = (acc0[34]+(alu55*val4));
    acc0[35] = (acc0[35]+(alu55*val5));
    acc0[36] = (acc0[36]+(alu55*val6));
    acc0[37] = (acc0[37]+(alu55*val7));
    acc0[38] = (acc0[38]+(alu55*val8));
    acc0[39] = (acc0[39]+(alu55*val9));
    acc0[40] = (acc0[40]+(alu55*val10));
    acc0[41] = (acc0[41]+(alu55*val11));
    acc0[42] = (acc0[42]+(alu55*val12));
    acc0[43] = (acc0[43]+(alu55*val13));
    acc0[44] = (acc0[44]+(alu55*val14));
    acc0[45] = (acc0[45]+(alu55*val15));
    acc0[46] = (acc0[46]+(alu55*val16));
    acc0[47] = (acc0[47]+(alu55*val17));
  }
  var val22 = data4_768[lidx0];
  var val23 = data4_768[(lidx0+16)];
  var val24 = data4_768[(lidx0+32)];
  var val25 = data4_768[(lidx0+48)];
  var val26 = data4_768[(lidx0+64)];
  var val27 = data4_768[(lidx0+80)];
  var val28 = data4_768[(lidx0+96)];
  var val29 = data4_768[(lidx0+112)];
  var val30 = data4_768[(lidx0+128)];
  var val31 = data4_768[(lidx0+144)];
  var val32 = data4_768[(lidx0+160)];
  var val33 = data4_768[(lidx0+176)];
  var val34 = data4_768[(lidx0+192)];
  var val35 = data4_768[(lidx0+208)];
  var val36 = data4_768[(lidx0+224)];
  var val37 = data4_768[(lidx0+240)];
  var alu105 = (lidx0+alu0);
  data0_76800[alu105] = (acc0[0]+val22);
  data0_76800[(alu105+16)] = (acc0[1]+val23);
  data0_76800[(alu105+32)] = (acc0[2]+val24);
  data0_76800[(alu105+48)] = (acc0[3]+val25);
  data0_76800[(alu105+64)] = (acc0[4]+val26);
  data0_76800[(alu105+80)] = (acc0[5]+val27);
  data0_76800[(alu105+96)] = (acc0[6]+val28);
  data0_76800[(alu105+112)] = (acc0[7]+val29);
  data0_76800[(alu105+128)] = (acc0[8]+val30);
  data0_76800[(alu105+144)] = (acc0[9]+val31);
  data0_76800[(alu105+160)] = (acc0[10]+val32);
  data0_76800[(alu105+176)] = (acc0[11]+val33);
  data0_76800[(alu105+192)] = (acc0[12]+val34);
  data0_76800[(alu105+208)] = (acc0[13]+val35);
  data0_76800[(alu105+224)] = (acc0[14]+val36);
  data0_76800[(alu105+240)] = (acc0[15]+val37);
  data0_76800[(alu105+256)] = (acc0[16]+val22);
  data0_76800[(alu105+272)] = (acc0[17]+val23);
  data0_76800[(alu105+288)] = (acc0[18]+val24);
  data0_76800[(alu105+304)] = (acc0[19]+val25);
  data0_76800[(alu105+320)] = (acc0[20]+val26);
  data0_76800[(alu105+336)] = (acc0[21]+val27);
  data0_76800[(alu105+352)] = (acc0[22]+val28);
  data0_76800[(alu105+368)] = (acc0[23]+val29);
  data0_76800[(alu105+384)] = (acc0[24]+val30);
  data0_76800[(alu105+400)] = (acc0[25]+val31);
  data0_76800[(alu105+416)] = (acc0[26]+val32);
  data0_76800[(alu105+432)] = (acc0[27]+val33);
  data0_76800[(alu105+448)] = (acc0[28]+val34);
  data0_76800[(alu105+464)] = (acc0[29]+val35);
  data0_76800[(alu105+480)] = (acc0[30]+val36);
  data0_76800[(alu105+496)] = (acc0[31]+val37);
  data0_76800[(alu105+512)] = (acc0[32]+val22);
  data0_76800[(alu105+528)] = (acc0[33]+val23);
  data0_76800[(alu105+544)] = (acc0[34]+val24);
  data0_76800[(alu105+560)] = (acc0[35]+val25);
  data0_76800[(alu105+576)] = (acc0[36]+val26);
  data0_76800[(alu105+592)] = (acc0[37]+val27);
  data0_76800[(alu105+608)] = (acc0[38]+val28);
  data0_76800[(alu105+624)] = (acc0[39]+val29);
  data0_76800[(alu105+640)] = (acc0[40]+val30);
  data0_76800[(alu105+656)] = (acc0[41]+val31);
  data0_76800[(alu105+672)] = (acc0[42]+val32);
  data0_76800[(alu105+688)] = (acc0[43]+val33);
  data0_76800[(alu105+704)] = (acc0[44]+val34);
  data0_76800[(alu105+720)] = (acc0[45]+val35);
  data0_76800[(alu105+736)] = (acc0[46]+val36);
  data0_76800[(alu105+752)] = (acc0[47]+val37);
}`;

const r_100_16_3_16_256n1 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_76800:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_76800:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_76800:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_196608:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_768:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,48>;
  var gidx0 = i32(gindex.x); /* 100 */
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
    var val0 = data1_76800[alu49];
    var val1 = data2_76800[alu49];
    var alu50 = (bitcast<i32>((bitcast<u32>(lidx0)<<8u))+Ridx0);
    var val2 = data3_196608[(alu50+65536)];
    var val3 = data3_196608[(alu50+69632)];
    var val4 = data3_196608[(alu50+73728)];
    var val5 = data3_196608[(alu50+77824)];
    var val6 = data3_196608[(alu50+81920)];
    var val7 = data3_196608[(alu50+86016)];
    var val8 = data3_196608[(alu50+90112)];
    var val9 = data3_196608[(alu50+94208)];
    var val10 = data3_196608[(alu50+98304)];
    var val11 = data3_196608[(alu50+102400)];
    var val12 = data3_196608[(alu50+106496)];
    var val13 = data3_196608[(alu50+110592)];
    var val14 = data3_196608[(alu50+114688)];
    var val15 = data3_196608[(alu50+118784)];
    var val16 = data3_196608[(alu50+122880)];
    var val17 = data3_196608[(alu50+126976)];
    var alu51 = (alu49+256);
    var val18 = data1_76800[alu51];
    var val19 = data2_76800[alu51];
    var alu52 = (alu49+512);
    var val20 = data1_76800[alu52];
    var val21 = data2_76800[alu52];
    var alu53 = (val0+val1);
    var alu54 = (val18+val19);
    var alu55 = (val20+val21);
    acc0[0] = (acc0[0]+(alu53*val2));
    acc0[1] = (acc0[1]+(alu53*val3));
    acc0[2] = (acc0[2]+(alu53*val4));
    acc0[3] = (acc0[3]+(alu53*val5));
    acc0[4] = (acc0[4]+(alu53*val6));
    acc0[5] = (acc0[5]+(alu53*val7));
    acc0[6] = (acc0[6]+(alu53*val8));
    acc0[7] = (acc0[7]+(alu53*val9));
    acc0[8] = (acc0[8]+(alu53*val10));
    acc0[9] = (acc0[9]+(alu53*val11));
    acc0[10] = (acc0[10]+(alu53*val12));
    acc0[11] = (acc0[11]+(alu53*val13));
    acc0[12] = (acc0[12]+(alu53*val14));
    acc0[13] = (acc0[13]+(alu53*val15));
    acc0[14] = (acc0[14]+(alu53*val16));
    acc0[15] = (acc0[15]+(alu53*val17));
    acc0[16] = (acc0[16]+(alu54*val2));
    acc0[17] = (acc0[17]+(alu54*val3));
    acc0[18] = (acc0[18]+(alu54*val4));
    acc0[19] = (acc0[19]+(alu54*val5));
    acc0[20] = (acc0[20]+(alu54*val6));
    acc0[21] = (acc0[21]+(alu54*val7));
    acc0[22] = (acc0[22]+(alu54*val8));
    acc0[23] = (acc0[23]+(alu54*val9));
    acc0[24] = (acc0[24]+(alu54*val10));
    acc0[25] = (acc0[25]+(alu54*val11));
    acc0[26] = (acc0[26]+(alu54*val12));
    acc0[27] = (acc0[27]+(alu54*val13));
    acc0[28] = (acc0[28]+(alu54*val14));
    acc0[29] = (acc0[29]+(alu54*val15));
    acc0[30] = (acc0[30]+(alu54*val16));
    acc0[31] = (acc0[31]+(alu54*val17));
    acc0[32] = (acc0[32]+(alu55*val2));
    acc0[33] = (acc0[33]+(alu55*val3));
    acc0[34] = (acc0[34]+(alu55*val4));
    acc0[35] = (acc0[35]+(alu55*val5));
    acc0[36] = (acc0[36]+(alu55*val6));
    acc0[37] = (acc0[37]+(alu55*val7));
    acc0[38] = (acc0[38]+(alu55*val8));
    acc0[39] = (acc0[39]+(alu55*val9));
    acc0[40] = (acc0[40]+(alu55*val10));
    acc0[41] = (acc0[41]+(alu55*val11));
    acc0[42] = (acc0[42]+(alu55*val12));
    acc0[43] = (acc0[43]+(alu55*val13));
    acc0[44] = (acc0[44]+(alu55*val14));
    acc0[45] = (acc0[45]+(alu55*val15));
    acc0[46] = (acc0[46]+(alu55*val16));
    acc0[47] = (acc0[47]+(alu55*val17));
  }
  var val22 = data4_768[(lidx0+256)];
  var val23 = data4_768[(lidx0+272)];
  var val24 = data4_768[(lidx0+288)];
  var val25 = data4_768[(lidx0+304)];
  var val26 = data4_768[(lidx0+320)];
  var val27 = data4_768[(lidx0+336)];
  var val28 = data4_768[(lidx0+352)];
  var val29 = data4_768[(lidx0+368)];
  var val30 = data4_768[(lidx0+384)];
  var val31 = data4_768[(lidx0+400)];
  var val32 = data4_768[(lidx0+416)];
  var val33 = data4_768[(lidx0+432)];
  var val34 = data4_768[(lidx0+448)];
  var val35 = data4_768[(lidx0+464)];
  var val36 = data4_768[(lidx0+480)];
  var val37 = data4_768[(lidx0+496)];
  var alu105 = (lidx0+alu0);
  data0_76800[alu105] = (acc0[0]+val22);
  data0_76800[(alu105+16)] = (acc0[1]+val23);
  data0_76800[(alu105+32)] = (acc0[2]+val24);
  data0_76800[(alu105+48)] = (acc0[3]+val25);
  data0_76800[(alu105+64)] = (acc0[4]+val26);
  data0_76800[(alu105+80)] = (acc0[5]+val27);
  data0_76800[(alu105+96)] = (acc0[6]+val28);
  data0_76800[(alu105+112)] = (acc0[7]+val29);
  data0_76800[(alu105+128)] = (acc0[8]+val30);
  data0_76800[(alu105+144)] = (acc0[9]+val31);
  data0_76800[(alu105+160)] = (acc0[10]+val32);
  data0_76800[(alu105+176)] = (acc0[11]+val33);
  data0_76800[(alu105+192)] = (acc0[12]+val34);
  data0_76800[(alu105+208)] = (acc0[13]+val35);
  data0_76800[(alu105+224)] = (acc0[14]+val36);
  data0_76800[(alu105+240)] = (acc0[15]+val37);
  data0_76800[(alu105+256)] = (acc0[16]+val22);
  data0_76800[(alu105+272)] = (acc0[17]+val23);
  data0_76800[(alu105+288)] = (acc0[18]+val24);
  data0_76800[(alu105+304)] = (acc0[19]+val25);
  data0_76800[(alu105+320)] = (acc0[20]+val26);
  data0_76800[(alu105+336)] = (acc0[21]+val27);
  data0_76800[(alu105+352)] = (acc0[22]+val28);
  data0_76800[(alu105+368)] = (acc0[23]+val29);
  data0_76800[(alu105+384)] = (acc0[24]+val30);
  data0_76800[(alu105+400)] = (acc0[25]+val31);
  data0_76800[(alu105+416)] = (acc0[26]+val32);
  data0_76800[(alu105+432)] = (acc0[27]+val33);
  data0_76800[(alu105+448)] = (acc0[28]+val34);
  data0_76800[(alu105+464)] = (acc0[29]+val35);
  data0_76800[(alu105+480)] = (acc0[30]+val36);
  data0_76800[(alu105+496)] = (acc0[31]+val37);
  data0_76800[(alu105+512)] = (acc0[32]+val22);
  data0_76800[(alu105+528)] = (acc0[33]+val23);
  data0_76800[(alu105+544)] = (acc0[34]+val24);
  data0_76800[(alu105+560)] = (acc0[35]+val25);
  data0_76800[(alu105+576)] = (acc0[36]+val26);
  data0_76800[(alu105+592)] = (acc0[37]+val27);
  data0_76800[(alu105+608)] = (acc0[38]+val28);
  data0_76800[(alu105+624)] = (acc0[39]+val29);
  data0_76800[(alu105+640)] = (acc0[40]+val30);
  data0_76800[(alu105+656)] = (acc0[41]+val31);
  data0_76800[(alu105+672)] = (acc0[42]+val32);
  data0_76800[(alu105+688)] = (acc0[43]+val33);
  data0_76800[(alu105+704)] = (acc0[44]+val34);
  data0_76800[(alu105+720)] = (acc0[45]+val35);
  data0_76800[(alu105+736)] = (acc0[46]+val36);
  data0_76800[(alu105+752)] = (acc0[47]+val37);
}`;

const r_75_16_16_4_256 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_76800:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_76800:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_196608:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_768:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 75 */
  var lidx0 = i32(lindex.x); /* 16 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<i32>((bitcast<u32>(gidx1)<<10u));
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 256; Ridx0++) {
    var alu4 = (cast1+Ridx0);
    var val0 = data1_76800[alu4];
    var val1 = data2_196608[(bitcast<i32>((cast0<<12u))+bitcast<i32>((bitcast<u32>(lidx0)<<8u))+Ridx0+131072)];
    var val2 = data1_76800[(alu4+256)];
    var val3 = data1_76800[(alu4+512)];
    var val4 = data1_76800[(alu4+768)];
    acc0[0] = (acc0[0]+(val0*val1));
    acc0[1] = (acc0[1]+(val2*val1));
    acc0[2] = (acc0[2]+(val3*val1));
    acc0[3] = (acc0[3]+(val4*val1));
  }
  var alu10 = (lidx0+bitcast<i32>((cast0<<4u)));
  var val5 = data3_768[(alu10+512)];
  var alu11 = (alu10+cast1);
  data0_76800[alu11] = (acc0[0]+val5);
  data0_76800[(alu11+256)] = (acc0[1]+val5);
  data0_76800[(alu11+512)] = (acc0[2]+val5);
  data0_76800[(alu11+768)] = (acc0[3]+val5);
}`;

const E_2_300_8_8_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_153600:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_76800:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_300:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_300:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_256:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_256:array<f32>;
@compute @workgroup_size(8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 8 */
  var gidx1 = i32(gindex.y); /* 300 */
  var lidx0 = i32(lindex.x); /* 8 */
  var alu0 = (lidx0+bitcast<i32>((bitcast<u32>(gidx0)<<5u)));
  var alu1 = (alu0+bitcast<i32>((bitcast<u32>(gidx1)<<8u)));
  var val0 = data1_76800[alu1];
  var val1 = data2_300[gidx1];
  var val2 = data3_300[gidx1];
  var val3 = data4_256[alu0];
  var val4 = data5_256[alu0];
  var val5 = data1_76800[(alu1+8)];
  var alu2 = (alu0+8);
  var val6 = data4_256[alu2];
  var alu3 = (alu0+16);
  var val7 = data4_256[alu3];
  var val8 = data5_256[alu2];
  var val9 = data1_76800[(alu1+16)];
  var alu4 = (alu0+24);
  var val10 = data4_256[alu4];
  var val11 = data5_256[alu3];
  var val12 = data1_76800[(alu1+24)];
  var val13 = data5_256[alu4];
  var gidx2 = i32(gindex.z); /* 2 */
  var alu5 = (alu1+(gidx2*76800));
  var alu6 = (gidx2<1);
  var alu7 = select(0.0f,(((val0-val1)*val2*val3)+val4),alu6);
  var alu8 = select(0.0f,(((val5-val1)*val2*val6)+val8),alu6);
  var alu9 = select(0.0f,(((val9-val1)*val2*val7)+val11),alu6);
  var alu10 = select(0.0f,(((val12-val1)*val2*val10)+val13),alu6);
  data0_153600[alu5] = alu7;
  data0_153600[(alu5+8)] = alu8;
  data0_153600[(alu5+16)] = alu9;
  data0_153600[(alu5+24)] = alu10;
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
  var alu2 = select((((val0-val1)*val2*val3)+val4),0.0f,(gidx2<1));
  data0_153600[(alu1+(gidx2*76800))] = alu2;
}`;

const E_9600_16 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_153600:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_153600:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_153600:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 9600 */
  var lidx0 = i32(lindex.x); /* 16 */
  var alu0 = (lidx0+bitcast<i32>((bitcast<u32>(gidx0)<<4u)));
  var val0 = data1_153600[alu0];
  var val1 = data2_153600[alu0];
  data0_153600[alu0] = (val0+val1);
}`;

const r_300_7_13_256 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_27300:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_153600:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_23296:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_91:array<f32>;
@compute @workgroup_size(13) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 7 */
  var gidx1 = i32(gindex.y); /* 300 */
  var lidx0 = i32(lindex.x); /* 13 */
  acc0[0] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 256; Ridx0++) {
    var val0 = data1_153600[(bitcast<i32>((bitcast<u32>(gidx1)<<8u))+Ridx0+76800)];
    var val1 = data2_23296[((gidx0*3328)+bitcast<i32>((bitcast<u32>(lidx0)<<8u))+Ridx0)];
    acc0[0] = (acc0[0]+(val0*val1));
  }
  var alu3 = (lidx0+(gidx0*13));
  var val2 = data3_91[alu3];
  data0_27300[(alu3+(gidx1*91))] = (1/(1.0f+exp2(((acc0[0]+val2)*-1.4426950408889634f))));
}`;

const r_200_16_16_3_256 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_153600:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_153600:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_65536:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_256:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,3>;
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 200 */
  var lidx0 = i32(lindex.x); /* 16 */
  var cast0 = bitcast<u32>(gidx0);
  var alu0 = (gidx1*768);
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 256; Ridx0++) {
    var alu4 = (alu0+Ridx0);
    var val0 = data1_153600[alu4];
    var val1 = data2_65536[(bitcast<i32>((cast0<<12u))+bitcast<i32>((bitcast<u32>(lidx0)<<8u))+Ridx0)];
    var val2 = data1_153600[(alu4+256)];
    var val3 = data1_153600[(alu4+512)];
    acc0[0] = (acc0[0]+(val0*val1));
    acc0[1] = (acc0[1]+(val2*val1));
    acc0[2] = (acc0[2]+(val3*val1));
  }
  var alu9 = (lidx0+bitcast<i32>((cast0<<4u)));
  var val4 = data3_256[alu9];
  var alu10 = (alu9+alu0);
  var alu11 = (acc0[0]+val4);
  var alu12 = (acc0[1]+val4);
  var alu13 = (acc0[2]+val4);
  var alu14 = select(0.0f,alu11,(0.0f<alu11));
  var alu15 = select(0.0f,alu12,(0.0f<alu12));
  var alu16 = select(0.0f,alu13,(0.0f<alu13));
  data0_153600[alu10] = alu14;
  data0_153600[(alu10+256)] = alu15;
  data0_153600[(alu10+512)] = alu16;
}`;

const E_2_4_2_4_32_2_2_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_32768:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_27300:array<f32>;
@compute @workgroup_size(4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 1024 */
  var gidx1 = i32(gindex.y); /* 4 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 4 */
  var cast0 = bitcast<u32>(gidx1);
  var cast1 = bitcast<u32>(gidx2);
  var cast2 = bitcast<u32>(lidx0);
  var alu0 = (gidx0>>2u);
  var alu1 = ((gidx0>>7u)&3);
  var alu2 = (lidx0+bitcast<i32>((cast1<<2u))+bitcast<i32>((bitcast<u32>((alu0&31))<<2u))+(alu1*37)+(gidx1*23));
  var alu3 = (gidx0&1);
  var alu4 = (gidx0>>9u);
  var alu5 = (alu4*57);
  var alu6 = (alu2+alu3+alu5);
  var alu7 = ((gidx2*180)+(lidx0*45)+alu1+(gidx1*11)+(alu4*5));
  var alu8 = ((alu6*361)>>15u);
  var alu9 = (alu7+alu8);
  var alu10 = ((bitcast<i32>((cast1<<12u))+alu0+bitcast<i32>((cast2<<10u))+bitcast<i32>((cast0<<8u)))<6825);
  var val0 = select(0.0f, data1_27300[(((alu9-(300*((alu9*219)>>16u)))*91)+(alu6-(91*alu8)))], alu10);
  var alu11 = (alu2+(alu3*90)+alu5+3);
  var alu12 = ((alu11*721)>>16u);
  var alu13 = (alu7+(alu12-alu3));
  var alu14 = select(0,1,(alu13<0));
  var val1 = select(0.0f, data1_27300[(((alu13-(300*(((alu13*219)>>16u)+alu14)))*91)+(alu11-(91*alu12)))], alu10);
  var alu15 = select(0.0f,1.0f,alu10);
  var alu16 = select((f32(-INFINITY)),0.0f,(alu15!=0.0f));
  var alu17 = (((gidx0>>1u)&1)<1);
  var alu18 = select(0.0f,(val0+alu16),alu17);
  var alu19 = select((val1+alu16),0.0f,alu17);
  data0_32768[(gidx0+bitcast<i32>((cast1<<14u))+bitcast<i32>((cast2<<12u))+bitcast<i32>((cast0<<10u)))] = (alu18+alu19);
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

const E_8192_2_2 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_32768:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_32768:array<f32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx1 = i32(gindex.y); /* 8192 */
  var cast0 = bitcast<i32>((bitcast<u32>(gidx1)<<2u));
  var val0 = data1_32768[cast0];
  var val1 = data1_32768[(cast0+1)];
  var val2 = data1_32768[(cast0+2)];
  var val3 = data1_32768[(cast0+3)];
  var gidx0 = i32(gindex.x); /* 2 */
  var alu0 = (gidx0+cast0);
  var alu1 = -val0;
  var alu2 = -val1;
  var alu3 = -val2;
  var alu4 = -val3;
  var alu5 = select(alu1,alu2,(alu1<alu2));
  var alu6 = select(alu3,alu4,(alu3<alu4));
  var alu7 = (gidx0<1);
  var alu8 = select(val0,val1,(val0<val1));
  var alu9 = select(0.0f,alu8,alu7);
  var alu10 = select(-alu5,0.0f,alu7);
  var alu11 = select(val2,val3,(val2<val3));
  var alu12 = select(0.0f,alu11,alu7);
  var alu13 = select(-alu6,0.0f,alu7);
  data0_32768[alu0] = (alu9+alu10);
  data0_32768[(alu0+2)] = (alu12+alu13);
}`;

const r_600_4_16_16 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32,16>;
@group(0) @binding(1)var<storage,read_write>data0_2400:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_153600:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_1024:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_4:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,1>;
  var acc1: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 600 */
  var lidx0 = i32(lindex.x); /* 16 */
  var cast0 = bitcast<u32>(gidx1);
  acc0[0] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 16; Ridx0++) {
    var alu1 = (lidx0+bitcast<i32>((bitcast<u32>(Ridx0)<<4u)));
    var val0 = data1_153600[(alu1+bitcast<i32>((cast0<<8u)))];
    var val1 = data2_1024[(alu1+bitcast<i32>((bitcast<u32>(gidx0)<<8u)))];
    acc0[0] = (acc0[0]+(val0*val1));
  }
  temp0[lidx0] = acc0[0];
  workgroupBarrier();
  acc1[0] = 0.0f;
  for (var Ridx104 = 0; Ridx104 < 16; Ridx104++) {
    var val2 = temp0[Ridx104];
    acc1[0] = (acc1[0]+val2);
  }
  var val3 = data3_4[gidx0];
  var alu9 = (lidx0==0);
  if (alu9) {
    data0_2400[(gidx0+bitcast<i32>((cast0<<2u)))] = (acc1[0]+val3);
  }
}`;

const E_1024_2_2_2_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_32768:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_32768:array<f32>;
@compute @workgroup_size(4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx2 = i32(gindex.z); /* 1024 */
  var lidx0 = i32(lindex.x); /* 4 */
  var cast0 = bitcast<u32>(gidx2);
  var cast1 = bitcast<u32>(lidx0);
  var alu0 = (gidx0>>1u);
  var alu1 = -alu0;
  var alu2 = (gidx0&1);
  var val0 = select(0.0f, data1_32768[(bitcast<i32>((bitcast<u32>(((bitcast<i32>((cast0<<4u))+bitcast<i32>((cast1<<2u))+alu0+3)&16383))<<1u))+alu2)], (-1<alu1));
  var cast2 = bitcast<i32>((cast0<<5u));
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

const E_4096_2_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_32768:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_32768:array<f32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx1 = i32(gindex.y); /* 4096 */
  var cast0 = bitcast<i32>((bitcast<u32>(gidx1)<<3u));
  var val0 = data1_32768[cast0];
  var val1 = data1_32768[(cast0+1)];
  var val2 = data1_32768[(cast0+2)];
  var val3 = data1_32768[(cast0+3)];
  var val4 = data1_32768[(cast0+4)];
  var val5 = data1_32768[(cast0+5)];
  var val6 = data1_32768[(cast0+6)];
  var val7 = data1_32768[(cast0+7)];
  var gidx0 = i32(gindex.x); /* 2 */
  var alu0 = (bitcast<i32>((bitcast<u32>(gidx0)<<2u))+cast0);
  var alu1 = -val0;
  var alu2 = -val1;
  var alu3 = -val2;
  var alu4 = -val3;
  var alu5 = -val4;
  var alu6 = -val5;
  var alu7 = -val6;
  var alu8 = -val7;
  var alu9 = select(alu1,alu5,(alu1<alu5));
  var alu10 = select(alu2,alu6,(alu2<alu6));
  var alu11 = select(alu3,alu7,(alu3<alu7));
  var alu12 = select(alu4,alu8,(alu4<alu8));
  var alu13 = (gidx0<1);
  var alu14 = select(val0,val4,(val0<val4));
  var alu15 = select(0.0f,alu14,alu13);
  var alu16 = select(-alu9,0.0f,alu13);
  var alu17 = select(val1,val5,(val1<val5));
  var alu18 = select(0.0f,alu17,alu13);
  var alu19 = select(-alu10,0.0f,alu13);
  var alu20 = select(val2,val6,(val2<val6));
  var alu21 = select(0.0f,alu20,alu13);
  var alu22 = select(-alu11,0.0f,alu13);
  var alu23 = select(val3,val7,(val3<val7));
  var alu24 = select(0.0f,alu23,alu13);
  var alu25 = select(-alu12,0.0f,alu13);
  data0_32768[alu0] = (alu15+alu16);
  data0_32768[(alu0+1)] = (alu18+alu19);
  data0_32768[(alu0+2)] = (alu21+alu22);
  data0_32768[(alu0+3)] = (alu24+alu25);
}`;

const E_1024_2_8_2 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_32768:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_32768:array<f32>;
@compute @workgroup_size(2) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 8 */
  var gidx2 = i32(gindex.z); /* 1024 */
  var cast0 = bitcast<u32>(gidx2);
  var cast1 = bitcast<i32>((cast0<<5u));
  var alu0 = (gidx0+cast1);
  var val0 = data1_32768[alu0];
  var gidx1 = i32(gindex.y); /* 2 */
  var alu1 = -gidx1;
  var val1 = select(0.0f, data1_32768[(gidx0+bitcast<i32>((bitcast<u32>(((gidx1+bitcast<i32>((cast0<<2u))+3)&4095))<<3u)))], (-1<alu1));
  var alu2 = (cast1-gidx0);
  var val2 = data1_32768[(alu2+15)];
  var val3 = data1_32768[(alu2+23)];
  var lidx0 = i32(lindex.x); /* 2 */
  var alu3 = (gidx1<1);
  var alu4 = select(0.0f,val0,alu3);
  var alu5 = select(val2,0.0f,alu3);
  var alu6 = select(0.0f,val3,(alu1<0));
  var alu7 = (lidx0<1);
  var alu8 = select(0.0f,(alu4+alu5),alu7);
  var alu9 = select((alu6+val1),0.0f,alu7);
  data0_32768[(alu0+bitcast<i32>((bitcast<u32>(gidx1)<<3u))+bitcast<i32>((bitcast<u32>(lidx0)<<4u)))] = (alu8+alu9);
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

const E_512_2_2_4_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_32768:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_32768:array<f32>;
@compute @workgroup_size(4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 8 */
  var gidx2 = i32(gindex.z); /* 512 */
  var lidx0 = i32(lindex.x); /* 4 */
  var cast0 = bitcast<u32>(gidx2);
  var cast1 = bitcast<i32>((cast0<<6u));
  var alu0 = (gidx0&3);
  var alu1 = (lidx0+bitcast<i32>((bitcast<u32>(alu0)<<2u)));
  var val0 = data1_32768[(alu1+cast1)];
  var alu2 = (gidx0>>2u);
  var alu3 = -alu2;
  var val1 = select(0.0f, data1_32768[(alu1+bitcast<i32>((bitcast<u32>(((bitcast<i32>((cast0<<2u))+alu2+3)&2047))<<4u)))], (-1<alu3));
  var alu4 = (((alu0*-4)-lidx0)+cast1);
  var val2 = data1_32768[(alu4+31)];
  var val3 = data1_32768[(alu4+47)];
  var gidx1 = i32(gindex.y); /* 2 */
  var alu5 = (gidx0<4);
  var alu6 = select(0.0f,val0,alu5);
  var alu7 = select(val2,0.0f,alu5);
  var alu8 = select(0.0f,val3,(alu3<0));
  var alu9 = (gidx1<1);
  var alu10 = select(0.0f,(alu6+alu7),alu9);
  var alu11 = select((alu8+val1),0.0f,alu9);
  data0_32768[(lidx0+bitcast<i32>((bitcast<u32>(gidx0)<<2u))+cast1+bitcast<i32>((bitcast<u32>(gidx1)<<5u)))] = (alu10+alu11);
}`;

const E_1024_2_16 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_32768:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_32768:array<f32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx1 = i32(gindex.y); /* 1024 */
  var cast0 = bitcast<i32>((bitcast<u32>(gidx1)<<5u));
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
  var val16 = data1_32768[(cast0+16)];
  var val17 = data1_32768[(cast0+17)];
  var val18 = data1_32768[(cast0+18)];
  var val19 = data1_32768[(cast0+19)];
  var val20 = data1_32768[(cast0+20)];
  var val21 = data1_32768[(cast0+21)];
  var val22 = data1_32768[(cast0+22)];
  var val23 = data1_32768[(cast0+23)];
  var val24 = data1_32768[(cast0+24)];
  var val25 = data1_32768[(cast0+25)];
  var val26 = data1_32768[(cast0+26)];
  var val27 = data1_32768[(cast0+27)];
  var val28 = data1_32768[(cast0+28)];
  var val29 = data1_32768[(cast0+29)];
  var val30 = data1_32768[(cast0+30)];
  var val31 = data1_32768[(cast0+31)];
  var gidx0 = i32(gindex.x); /* 2 */
  var alu0 = (bitcast<i32>((bitcast<u32>(gidx0)<<4u))+cast0);
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
  var alu33 = select(alu1,alu17,(alu1<alu17));
  var alu34 = select(alu2,alu18,(alu2<alu18));
  var alu35 = select(alu3,alu19,(alu3<alu19));
  var alu36 = select(alu4,alu20,(alu4<alu20));
  var alu37 = select(alu5,alu21,(alu5<alu21));
  var alu38 = select(alu6,alu22,(alu6<alu22));
  var alu39 = select(alu7,alu23,(alu7<alu23));
  var alu40 = select(alu8,alu24,(alu8<alu24));
  var alu41 = select(alu9,alu25,(alu9<alu25));
  var alu42 = select(alu10,alu26,(alu10<alu26));
  var alu43 = select(alu11,alu27,(alu11<alu27));
  var alu44 = select(alu12,alu28,(alu12<alu28));
  var alu45 = select(alu13,alu29,(alu13<alu29));
  var alu46 = select(alu14,alu30,(alu14<alu30));
  var alu47 = select(alu15,alu31,(alu15<alu31));
  var alu48 = select(alu16,alu32,(alu16<alu32));
  var alu49 = (gidx0<1);
  var alu50 = select(val0,val16,(val0<val16));
  var alu51 = select(0.0f,alu50,alu49);
  var alu52 = select(-alu33,0.0f,alu49);
  var alu53 = select(val1,val17,(val1<val17));
  var alu54 = select(0.0f,alu53,alu49);
  var alu55 = select(-alu34,0.0f,alu49);
  var alu56 = select(val2,val18,(val2<val18));
  var alu57 = select(0.0f,alu56,alu49);
  var alu58 = select(-alu35,0.0f,alu49);
  var alu59 = select(val3,val19,(val3<val19));
  var alu60 = select(0.0f,alu59,alu49);
  var alu61 = select(-alu36,0.0f,alu49);
  var alu62 = select(val4,val20,(val4<val20));
  var alu63 = select(0.0f,alu62,alu49);
  var alu64 = select(-alu37,0.0f,alu49);
  var alu65 = select(val5,val21,(val5<val21));
  var alu66 = select(0.0f,alu65,alu49);
  var alu67 = select(-alu38,0.0f,alu49);
  var alu68 = select(val6,val22,(val6<val22));
  var alu69 = select(0.0f,alu68,alu49);
  var alu70 = select(-alu39,0.0f,alu49);
  var alu71 = select(val7,val23,(val7<val23));
  var alu72 = select(0.0f,alu71,alu49);
  var alu73 = select(-alu40,0.0f,alu49);
  var alu74 = select(val8,val24,(val8<val24));
  var alu75 = select(0.0f,alu74,alu49);
  var alu76 = select(-alu41,0.0f,alu49);
  var alu77 = select(val9,val25,(val9<val25));
  var alu78 = select(0.0f,alu77,alu49);
  var alu79 = select(-alu42,0.0f,alu49);
  var alu80 = select(val10,val26,(val10<val26));
  var alu81 = select(0.0f,alu80,alu49);
  var alu82 = select(-alu43,0.0f,alu49);
  var alu83 = select(val11,val27,(val11<val27));
  var alu84 = select(0.0f,alu83,alu49);
  var alu85 = select(-alu44,0.0f,alu49);
  var alu86 = select(val12,val28,(val12<val28));
  var alu87 = select(0.0f,alu86,alu49);
  var alu88 = select(-alu45,0.0f,alu49);
  var alu89 = select(val13,val29,(val13<val29));
  var alu90 = select(0.0f,alu89,alu49);
  var alu91 = select(-alu46,0.0f,alu49);
  var alu92 = select(val14,val30,(val14<val30));
  var alu93 = select(0.0f,alu92,alu49);
  var alu94 = select(-alu47,0.0f,alu49);
  var alu95 = select(val15,val31,(val15<val31));
  var alu96 = select(0.0f,alu95,alu49);
  var alu97 = select(-alu48,0.0f,alu49);
  data0_32768[alu0] = (alu51+alu52);
  data0_32768[(alu0+1)] = (alu54+alu55);
  data0_32768[(alu0+2)] = (alu57+alu58);
  data0_32768[(alu0+3)] = (alu60+alu61);
  data0_32768[(alu0+4)] = (alu63+alu64);
  data0_32768[(alu0+5)] = (alu66+alu67);
  data0_32768[(alu0+6)] = (alu69+alu70);
  data0_32768[(alu0+7)] = (alu72+alu73);
  data0_32768[(alu0+8)] = (alu75+alu76);
  data0_32768[(alu0+9)] = (alu78+alu79);
  data0_32768[(alu0+10)] = (alu81+alu82);
  data0_32768[(alu0+11)] = (alu84+alu85);
  data0_32768[(alu0+12)] = (alu87+alu88);
  data0_32768[(alu0+13)] = (alu90+alu91);
  data0_32768[(alu0+14)] = (alu93+alu94);
  data0_32768[(alu0+15)] = (alu96+alu97);
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
  var cast1 = bitcast<u32>(lidx0);
  var alu0 = (gidx0>>5u);
  var alu1 = -alu0;
  var alu2 = (gidx0&31);
  var val0 = select(0.0f, data1_32768[(bitcast<i32>((bitcast<u32>(((bitcast<i32>((cast0<<4u))+bitcast<i32>((cast1<<2u))+alu0+3)&1023))<<5u))+alu2)], (-1<alu1));
  var cast2 = bitcast<i32>((cast0<<9u));
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

const E_64_2_32_8 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_32768:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_32768:array<f32>;
@compute @workgroup_size(8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 32 */
  var gidx2 = i32(gindex.z); /* 64 */
  var lidx0 = i32(lindex.x); /* 8 */
  var alu0 = (gidx0+bitcast<i32>((bitcast<u32>(gidx2)<<9u))+bitcast<i32>((bitcast<u32>(lidx0)<<6u)));
  var val0 = data1_32768[alu0];
  var val1 = data1_32768[(alu0+32)];
  var gidx1 = i32(gindex.y); /* 2 */
  var alu1 = -val0;
  var alu2 = -val1;
  var alu3 = select(alu1,alu2,(alu1<alu2));
  var alu4 = (gidx1<1);
  var alu5 = select(val0,val1,(val0<val1));
  var alu6 = select(0.0f,alu5,alu4);
  var alu7 = select(-alu3,0.0f,alu4);
  data0_32768[(alu0+bitcast<i32>((bitcast<u32>(gidx1)<<5u)))] = (alu6+alu7);
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

const E_64_2_128_2 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_32768:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_32768:array<f32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 128 */
  var gidx2 = i32(gindex.z); /* 64 */
  var cast0 = bitcast<u32>(gidx2);
  var cast1 = bitcast<i32>((cast0<<9u));
  var alu0 = (gidx0+cast1);
  var val0 = data1_32768[alu0];
  var gidx1 = i32(gindex.y); /* 2 */
  var alu1 = -gidx1;
  var val1 = select(0.0f, data1_32768[(gidx0+bitcast<i32>((bitcast<u32>(((gidx1+bitcast<i32>((cast0<<2u))+3)&255))<<7u)))], (-1<alu1));
  var alu2 = (cast1-gidx0);
  var val2 = data1_32768[(alu2+255)];
  var val3 = data1_32768[(alu2+383)];
  var alu3 = (alu0+bitcast<i32>((bitcast<u32>(gidx1)<<7u)));
  var alu4 = (gidx1<1);
  var alu5 = select(0.0f,val0,alu4);
  var alu6 = select(val2,0.0f,alu4);
  var alu7 = select(0.0f,val3,(alu1<0));
  data0_32768[alu3] = (alu5+alu6);
  data0_32768[(alu3+256)] = (alu7+val1);
}`;

const E_32_2_128_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_32768:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_32768:array<f32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 128 */
  var gidx2 = i32(gindex.z); /* 32 */
  var alu0 = (gidx0+bitcast<i32>((bitcast<u32>(gidx2)<<10u)));
  var val0 = data1_32768[alu0];
  var val1 = data1_32768[(alu0+128)];
  var val2 = data1_32768[(alu0+256)];
  var val3 = data1_32768[(alu0+384)];
  var val4 = data1_32768[(alu0+512)];
  var val5 = data1_32768[(alu0+640)];
  var val6 = data1_32768[(alu0+768)];
  var val7 = data1_32768[(alu0+896)];
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
  data0_32768[(alu1+256)] = (alu19+alu20);
  data0_32768[(alu1+512)] = (alu22+alu23);
  data0_32768[(alu1+768)] = (alu25+alu26);
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
  var cast1 = bitcast<i32>((cast0<<10u));
  var alu0 = (gidx0&31);
  var alu1 = (lidx0+bitcast<i32>((bitcast<u32>(alu0)<<3u)));
  var val0 = data1_32768[(alu1+cast1)];
  var alu2 = (gidx0>>5u);
  var alu3 = -alu2;
  var val1 = select(0.0f, data1_32768[(alu1+bitcast<i32>((bitcast<u32>(((bitcast<i32>((cast0<<2u))+alu2+3)&127))<<8u)))], (-1<alu3));
  var alu4 = (((alu0*-8)-lidx0)+cast1);
  var val2 = data1_32768[(alu4+511)];
  var val3 = data1_32768[(alu4+767)];
  var gidx1 = i32(gindex.y); /* 2 */
  var alu5 = (gidx0<32);
  var alu6 = select(0.0f,val0,alu5);
  var alu7 = select(val2,0.0f,alu5);
  var alu8 = select(0.0f,val3,(alu3<0));
  var alu9 = (gidx1<1);
  var alu10 = select(0.0f,(alu6+alu7),alu9);
  var alu11 = select((alu8+val1),0.0f,alu9);
  data0_32768[(lidx0+bitcast<i32>((bitcast<u32>(gidx0)<<3u))+cast1+bitcast<i32>((bitcast<u32>(gidx1)<<9u)))] = (alu10+alu11);
}`;

const E_16_2_256_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_32768:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_32768:array<f32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 256 */
  var gidx2 = i32(gindex.z); /* 16 */
  var alu0 = (gidx0+bitcast<i32>((bitcast<u32>(gidx2)<<11u)));
  var val0 = data1_32768[alu0];
  var val1 = data1_32768[(alu0+256)];
  var val2 = data1_32768[(alu0+512)];
  var val3 = data1_32768[(alu0+768)];
  var val4 = data1_32768[(alu0+1024)];
  var val5 = data1_32768[(alu0+1280)];
  var val6 = data1_32768[(alu0+1536)];
  var val7 = data1_32768[(alu0+1792)];
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
  data0_32768[(alu1+512)] = (alu19+alu20);
  data0_32768[(alu1+1024)] = (alu22+alu23);
  data0_32768[(alu1+1536)] = (alu25+alu26);
}`;

const E_4_2_2_512_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_32768:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_32768:array<f32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 1024 */
  var gidx2 = i32(gindex.z); /* 4 */
  var cast0 = bitcast<u32>(gidx2);
  var cast1 = bitcast<i32>((cast0<<13u));
  var alu0 = (gidx0&511);
  var alu1 = (cast1+alu0);
  var val0 = data1_32768[alu1];
  var alu2 = (gidx0>>9u);
  var alu3 = -alu2;
  var alu4 = (-1<alu3);
  var val1 = select(0.0f, data1_32768[(bitcast<i32>((bitcast<u32>(((bitcast<i32>((cast0<<4u))+alu2+15)&63))<<9u))+alu0)], alu4);
  var alu5 = (gidx0+cast1);
  var val2 = select(0.0f, data1_32768[(alu5+1536)], alu4);
  var val3 = select(0.0f, data1_32768[(alu5+3584)], alu4);
  var val4 = select(0.0f, data1_32768[(alu5+5632)], alu4);
  var val5 = data1_32768[(alu1+2048)];
  var val6 = data1_32768[(alu1+4096)];
  var val7 = data1_32768[(alu1+6144)];
  var alu6 = (cast1-alu0);
  var val8 = data1_32768[(alu6+1023)];
  var val9 = data1_32768[(alu6+1535)];
  var val10 = data1_32768[(alu6+3071)];
  var val11 = data1_32768[(alu6+3583)];
  var val12 = data1_32768[(alu6+5119)];
  var val13 = data1_32768[(alu6+5631)];
  var val14 = data1_32768[(alu6+7167)];
  var val15 = data1_32768[(alu6+7679)];
  var gidx1 = i32(gindex.y); /* 2 */
  var alu7 = (alu5+bitcast<i32>((bitcast<u32>(gidx1)<<10u)));
  var alu8 = (gidx0<512);
  var alu9 = select(0.0f,val0,alu8);
  var alu10 = select(val8,0.0f,alu8);
  var alu11 = select(0.0f,val5,alu8);
  var alu12 = select(val10,0.0f,alu8);
  var alu13 = select(0.0f,val6,alu8);
  var alu14 = select(val12,0.0f,alu8);
  var alu15 = select(0.0f,val7,alu8);
  var alu16 = select(val14,0.0f,alu8);
  var alu17 = (alu3<0);
  var alu18 = select(0.0f,val9,alu17);
  var alu19 = (gidx1<1);
  var alu20 = select(0.0f,(alu9+alu10),alu19);
  var alu21 = select((alu18+val2),0.0f,alu19);
  var alu22 = select(0.0f,val11,alu17);
  var alu23 = select(0.0f,(alu11+alu12),alu19);
  var alu24 = select((alu22+val3),0.0f,alu19);
  var alu25 = select(0.0f,val13,alu17);
  var alu26 = select(0.0f,(alu13+alu14),alu19);
  var alu27 = select((alu25+val4),0.0f,alu19);
  var alu28 = select(0.0f,val15,alu17);
  var alu29 = select(0.0f,(alu15+alu16),alu19);
  var alu30 = select((alu28+val1),0.0f,alu19);
  data0_32768[alu7] = (alu20+alu21);
  data0_32768[(alu7+2048)] = (alu23+alu24);
  data0_32768[(alu7+4096)] = (alu26+alu27);
  data0_32768[(alu7+6144)] = (alu29+alu30);
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

const E_2_2_1024_8 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_32768:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_32768:array<f32>;
@compute @workgroup_size(1) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 1024 */
  var val0 = data1_32768[gidx0];
  var val1 = data1_32768[(gidx0+4096)];
  var val2 = data1_32768[(gidx0+8192)];
  var val3 = data1_32768[(gidx0+12288)];
  var val4 = data1_32768[(gidx0+16384)];
  var val5 = data1_32768[(gidx0+20480)];
  var val6 = data1_32768[(gidx0+24576)];
  var val7 = data1_32768[(gidx0+28672)];
  var gidx1 = i32(gindex.y); /* 2 */
  var alu0 = -gidx1;
  var alu1 = (gidx0+bitcast<i32>((bitcast<u32>(gidx1)<<10u)));
  var alu2 = (-1<alu0);
  var val8 = select(0.0f, data1_32768[(alu1+3072)], alu2);
  var val9 = select(0.0f, data1_32768[(alu1+7168)], alu2);
  var val10 = select(0.0f, data1_32768[(alu1+11264)], alu2);
  var val11 = select(0.0f, data1_32768[(alu1+15360)], alu2);
  var val12 = select(0.0f, data1_32768[(alu1+19456)], alu2);
  var val13 = select(0.0f, data1_32768[(alu1+23552)], alu2);
  var val14 = select(0.0f, data1_32768[(alu1+27648)], alu2);
  var val15 = select(0.0f, data1_32768[(gidx0+(gidx1*-31744)+31744)], alu2);
  var val16 = data1_32768[(2047-gidx0)];
  var val17 = data1_32768[(3071-gidx0)];
  var val18 = data1_32768[(6143-gidx0)];
  var val19 = data1_32768[(7167-gidx0)];
  var val20 = data1_32768[(10239-gidx0)];
  var val21 = data1_32768[(11263-gidx0)];
  var val22 = data1_32768[(14335-gidx0)];
  var val23 = data1_32768[(15359-gidx0)];
  var val24 = data1_32768[(18431-gidx0)];
  var val25 = data1_32768[(19455-gidx0)];
  var val26 = data1_32768[(22527-gidx0)];
  var val27 = data1_32768[(23551-gidx0)];
  var val28 = data1_32768[(26623-gidx0)];
  var val29 = data1_32768[(27647-gidx0)];
  var val30 = data1_32768[(30719-gidx0)];
  var val31 = data1_32768[(31743-gidx0)];
  var gidx2 = i32(gindex.z); /* 2 */
  var alu3 = (alu1+bitcast<i32>((bitcast<u32>(gidx2)<<11u)));
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
  data0_32768[alu3] = (alu24+alu25);
  data0_32768[(alu3+4096)] = (alu27+alu28);
  data0_32768[(alu3+8192)] = (alu30+alu31);
  data0_32768[(alu3+12288)] = (alu33+alu34);
  data0_32768[(alu3+16384)] = (alu36+alu37);
  data0_32768[(alu3+20480)] = (alu39+alu40);
  data0_32768[(alu3+24576)] = (alu42+alu43);
  data0_32768[(alu3+28672)] = (alu45+alu46);
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
@compute @workgroup_size(4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 2048 */
  var lidx0 = i32(lindex.x); /* 4 */
  var alu0 = (lidx0+bitcast<i32>((bitcast<u32>(gidx0)<<2u)));
  var val0 = data1_32768[alu0];
  var gidx1 = i32(gindex.y); /* 2 */
  var alu1 = -gidx1;
  var val1 = select(0.0f, data1_32768[(alu0+(gidx1*-24576)+24576)], (-1<alu1));
  var alu2 = ((gidx0*-4)-lidx0);
  var val2 = data1_32768[(alu2+16383)];
  var val3 = data1_32768[(alu2+24575)];
  var gidx2 = i32(gindex.z); /* 2 */
  var alu3 = (gidx1<1);
  var alu4 = select(0.0f,val0,alu3);
  var alu5 = select(val2,0.0f,alu3);
  var alu6 = select(0.0f,val3,(alu1<0));
  var alu7 = (gidx2<1);
  var alu8 = select(0.0f,(alu4+alu5),alu7);
  var alu9 = select((alu6+val1),0.0f,alu7);
  data0_32768[(alu0+bitcast<i32>((bitcast<u32>(gidx1)<<13u))+bitcast<i32>((bitcast<u32>(gidx2)<<14u)))] = (alu8+alu9);
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
@compute @workgroup_size(4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 4096 */
  var lidx0 = i32(lindex.x); /* 4 */
  var alu0 = (lidx0+bitcast<i32>((bitcast<u32>(gidx0)<<2u)));
  var val0 = data1_32768[alu0];
  var val1 = data1_32768[(((gidx0*-4)-lidx0)+32767)];
  var gidx1 = i32(gindex.y); /* 2 */
  var alu1 = -val0;
  var alu2 = -val1;
  var alu3 = select(alu1,alu2,(alu1<alu2));
  var alu4 = (gidx1<1);
  var alu5 = select(val0,val1,(val0<val1));
  var alu6 = select(0.0f,alu5,alu4);
  var alu7 = select(-alu3,0.0f,alu4);
  data0_32768[(alu0+bitcast<i32>((bitcast<u32>(gidx1)<<14u)))] = (alu6+alu7);
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

const r_75_28_4_975 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<i32,112>;
@group(0) @binding(1)var<storage,read_write>data0_300:array<i32>;
@group(0) @binding(2)var<storage,read_write>data1_27300:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_32768:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_27300:array<i32>;
@group(0) @binding(5)var<storage,read_write>data4_27300:array<i32>;
@compute @workgroup_size(28,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<i32,1>;
  var acc1: array<i32,1>;
  var gidx0 = i32(gindex.x); /* 75 */
  var lidx1 = i32(lindex.y); /* 4 */
  var alu0 = (lidx1+bitcast<i32>((bitcast<u32>(gidx0)<<2u)));
  var val0 = data4_27300[alu0];
  var val1 = data2_32768[alu0];
  var lidx0 = i32(lindex.x); /* 28 */
  acc0[0] = 0;
  for (var Ridx0 = 0; Ridx0 < 975; Ridx0++) {
    var alu2 = ((lidx0*975)+Ridx0);
    var val2 = data3_27300[alu2];
    var val3 = data1_27300[alu2];
    acc0[0] = (acc0[0]+((i32(((val3==val1)&(val2==val0))))*alu2));
  }
  var alu5 = (lidx1*28);
  temp0[(lidx0+alu5)] = acc0[0];
  workgroupBarrier();
  acc1[0] = 0;
  for (var Ridx102 = 0; Ridx102 < 28; Ridx102++) {
    var val4 = temp0[(alu5+Ridx102)];
    acc1[0] = (acc1[0]+val4);
  }
  var alu11 = (lidx0==0);
  if (alu11) {
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
  var alu6 = (val0*384.0f);
  var alu7 = (val1*384.0f);
  var alu8 = (val2*384.0f);
  var alu9 = (val5*384.0f);
  var alu10 = select(alu6,0.0f,(alu6<0.0f));
  var alu11 = select(alu7,0.0f,(alu7<0.0f));
  var alu12 = select(alu8,0.0f,(alu8<0.0f));
  var alu13 = select(alu9,0.0f,(alu9<0.0f));
  var alu14 = select(alu10,384.0f,(384.0f<alu10));
  var alu15 = select(alu11,384.0f,(384.0f<alu11));
  var alu16 = select(alu12,384.0f,(384.0f<alu12));
  var alu17 = select(alu13,384.0f,(384.0f<alu13));
  data0_1800[alu1] = alu14;
  data0_1800[alu2] = alu15;
  data0_1800[alu3] = alu16;
  data0_1800[alu4] = val3;
  data0_1800[alu5] = val4;
  data0_1800[alu0] = alu17;
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

    const kernels = [E_4608_32_3, r_100_16_16_3_256, E_9216_16_3, E_384_24_16_3, E_384_12_16_3_2, r_577_24_8_8_2_16_2_3, r_2_2_145_16_24, r_2_2_145_32_12, E_2_145_24_16_2, r_116_24_16_5_384, r_116_24_16_5_384, r_116_24_16_5_384, r_4_6_5_145_29_64, r_435_8_145, r_435_8_145n1, r_4_145_16_2_3_4_145, r_2_2_29_12_16_4_5_2_96, r_580_16_24, r_580_16_24n1, E_290_24_16_2, r_20_32_29_4_4_3_384, r_5_32_29_4_3_4_1536, r_580_16_24, r_580_16_24n1, E_290_24_16_2, r_116_24_16_5_384, r_116_24_16_5_384, r_116_24_16_5_384, r_4_6_5_145_29_64, r_435_8_145, r_435_8_145n1, r_4_145_16_2_3_4_145, r_10_32_29_4_3_2_384, r_580_16_24, r_580_16_24n1, E_290_24_16_2, r_20_32_29_4_4_3_384, r_5_32_29_4_3_4_1536, r_580_16_24, r_580_16_24n1, E_290_24_16_2, r_116_24_16_5_384, r_116_24_16_5_384, r_116_24_16_5_384, r_4_6_5_145_29_64, r_435_8_145, r_435_8_145n1, r_4_145_16_2_3_4_145, r_10_32_29_4_3_2_384, r_580_16_24, r_580_16_24n1, E_290_24_16_2, r_20_32_29_4_4_3_384, r_5_32_29_4_3_4_1536, r_580_16_24n2, r_580_16_24, r_580_16_24n3, r_580_16_24n1, E_580_48_8, E_192_2_2_12_2_12, r_20_32_29_4_3_384, r_20_32_29_4_3_384, r_20_32_29_4_3_384, r_6_116_20_29_5_64, r_120_29_145_4, r_120_29_145_4n1, r_145_6_16_2_4_2_145_4, r_10_32_29_4_3_2_384n1, r_580_16_24, r_580_16_24n1, E_290_24_16_2, r_20_32_29_4_4_3_384, r_5_32_29_4_3_4_1536, r_580_16_24, r_580_16_24n1, E_290_24_16_2, r_116_24_16_5_384, r_116_24_16_5_384, r_116_24_16_5_384, r_4_6_5_145_29_64, r_435_8_145, r_435_8_145n1, r_4_145_16_2_3_4_145, r_10_32_29_4_3_2_384, r_580_16_24, r_580_16_24n1, E_290_24_16_2, r_20_32_29_4_4_3_384, r_5_32_29_4_3_4_1536, r_580_16_24, r_580_16_24n1, E_290_24_16_2, r_116_24_16_5_384, r_116_24_16_5_384, r_116_24_16_5_384, r_4_6_5_145_29_64, r_435_8_145, r_435_8_145n1, r_4_145_16_2_3_4_145, r_10_32_29_4_3_2_384, r_580_16_24, r_580_16_24n1, E_290_24_16_2, r_20_32_29_4_4_3_384, r_5_32_29_4_3_4_1536, r_580_16_24n2, r_580_16_24, r_580_16_24n3, r_580_16_24n1, E_580_48_8, E_192_2_2_12_2_12, r_20_32_29_4_3_384, r_20_32_29_4_3_384, r_20_32_29_4_3_384, r_6_116_20_29_5_64, r_120_29_145_4, r_120_29_145_4n1, r_145_6_16_2_4_2_145_4, r_10_32_29_4_3_2_384n1, r_580_16_24, r_580_16_24n1, E_290_24_16_2, r_20_32_29_4_4_3_384, r_5_32_29_4_3_4_1536, r_580_16_24, r_580_16_24n1, E_290_24_16_2, r_116_24_16_5_384, r_116_24_16_5_384, r_116_24_16_5_384, r_4_6_5_145_29_64, r_435_8_145, r_435_8_145n1, r_4_145_16_2_3_4_145, r_10_32_29_4_3_2_384, r_580_16_24, r_580_16_24n1, E_290_24_16_2, r_20_32_29_4_4_3_384, r_5_32_29_4_3_4_1536, r_580_16_24, r_580_16_24n1, E_290_24_16_2, r_116_24_16_5_384, r_116_24_16_5_384, r_116_24_16_5_384, r_4_6_5_145_29_64, r_435_8_145, r_435_8_145n1, r_4_145_16_2_3_4_145, r_10_32_29_4_3_2_384, r_580_16_24, r_580_16_24n1, E_290_24_16_2, r_20_32_29_4_4_3_384, r_5_32_29_4_3_4_1536, r_580_16_24n2, r_580_16_24, r_580_16_24n3, r_580_16_24n1, E_580_48_8, E_192_2_2_12_2_12, r_20_32_29_4_3_384, r_20_32_29_4_3_384, r_20_32_29_4_3_384, r_6_116_20_29_5_64, r_120_29_145_4, r_120_29_145_4n1, r_145_6_16_2_4_2_145_4, r_10_32_29_4_3_2_384n1, r_580_16_24, r_580_16_24n1, E_290_24_16_2, r_20_32_29_4_4_3_384, r_5_32_29_4_3_4_1536, r_580_16_24, r_580_16_24n1, E_290_24_16_2, r_116_24_16_5_384, r_116_24_16_5_384, r_116_24_16_5_384, r_4_6_5_145_29_64, r_435_8_145, r_435_8_145n1, r_4_145_16_2_3_4_145, r_10_32_29_4_3_2_384, r_580_16_24, r_580_16_24n1, E_290_24_16_2, r_20_32_29_4_4_3_384, r_5_32_29_4_3_4_1536, r_580_16_24, r_580_16_24n1, E_290_24_16_2, r_116_24_16_5_384, r_116_24_16_5_384, r_116_24_16_5_384, r_4_6_5_145_29_64, r_435_8_145, r_435_8_145n1, r_4_145_16_2_3_4_145, r_10_32_29_4_3_2_384, r_580_16_24, r_580_16_24n1, E_290_24_16_2, r_20_32_29_4_4_3_384, r_5_32_29_4_3_4_1536, r_580_16_24, r_580_16_24n1, E_192_2_2_12_2_12, E_13824_32_2, r_18_64_32_4_1536, r_144_4_256, E_192_32_8_3, r_576_32_8, E_256_24_8_3, r_6_6_16_4_4_8_128_3_3, r_576_32_4, E_36_128_16, r_576_32_4n1, E_8_576_16, r_12_8_16_2_4_6_128_3_3, r_576_32_4, E_36_128_16, r_576_32_4n1, E_8_576_16, r_12_8_16_2_4_6_128_3_3, r_576_32_4, E_36_128_16, r_576_32_4n1, E_8_576_16, r_12_8_16_2_4_6_128_3_3, r_576_32_4, E_36_128_16, r_576_32_4n1, E_8_576_16, r_12_8_16_2_4_6_128_3_3, r_576_32_4, E_36_128_16, r_576_32_4n1, E_8_576_16, r_12_8_16_2_4_6_128_3_3, r_576_32_4, E_36_128_16, r_576_32_4n1, E_40_144_16_4, r_18_256_32_160_4, r_144_4_256, E_192_32_8_3, r_576_32_8, E_576_16_16, r_144_4_256, E_192_32_8_3, r_576_32_8, E_72_64_8_4, r_8_24_16_2_12_2_256, r_12_16_8_4_2_3_4_24_24_4_64_4, r_8_24_16_2_12_2_256, r_144_4_256n1, r_576_16_16, E_72_64_8_4n1, r_96_91_3_2_256, r_18_64_32_4_256, E_256_2_2, r_576_16_36, r_18_64_32_4_256, E_512_2, r_36_4_16_256, E_128_2_2_2, E_256_2_2n1, E_512_2, E_64_2_4_2, E_128_4_2, E_256_2_2n1, E_512_2, E_32_2_8_2, E_64_8_2, E_128_4_2, E_256_2_2n1, E_512_2, E_2_2_16_16, E_32_16_2, E_64_8_2, E_128_4_2, E_256_2_2n1, E_512_2, E_2_2_32_8, E_2_32_16, E_32_16_2, E_64_8_2, E_128_4_2, E_256_2_2n1, E_512_2, E_2_2_64_4, E_2_64_8, E_2_32_16, E_32_16_2, E_64_8_2, E_128_4_2, E_256_2_2n1, E_512_2, E_2_2_128_2, E_2_128_4, E_2_64_8, E_2_32_16, E_32_16_2, E_64_8_2, E_128_4_2, E_256_2_2n1, E_512_2, E_2_256_2, E_2_256_2n1, E_2_128_4, E_2_64_8, E_2_32_16, E_32_16_2, E_64_8_2, E_128_4_2, E_256_2_2n1, E_512_2, E_512_2n1, E_2_256_2n1, E_2_128_4, E_2_64_8, E_2_32_16, E_32_16_2, E_64_8_2, E_128_4_2, E_256_2_2n1, E_512_2, r_576_16_36n1, r_576_16_36n2, r_75_4_24_4_24_24_24_4, r_60_64_32_4_5_8_2, r_100_8_16_8_3_2_32, r_30_8_8_4_5_2_256, r_75_16_8_4_2_4_64, r_300_100_8_4_3_8, r_300_8_75_4, r_150_8_2_75_4, r_2_8_100_16_3_300, r_60_2_16_8_5_256, r_300_16_16, r_300_16_16n1, E_300_32_8, r_300_2_2_16_16_16, r_300_2_16_256, r_2400_2_2, r_2400_2_2n1, r_300_8_16_2_2, r_100_16_16_3_256n1, r_300_16_16, r_300_16_16n1, E_300_32_8, r_30_64_16_5_2_2_256, r_20_4_16_2_5_3_2_2048, r_300_16_16, r_300_16_16n1, E_300_32_8, r_100_16_3_16_256, r_100_16_3_16_256n1, r_75_16_16_4_256, r_300_16_16, r_300_100_8_4_3_8, r_300_16_16n1, r_300_8_75_4, E_2_300_8_8_4, r_150_8_2_75_4, r_2_8_100_16_3_300, r_100_16_16_3_256n1, r_300_16_16, r_300_16_16n1, E_300_32_8, r_300_2_2_16_16_16, r_300_2_16_256, r_2400_2_2, r_2400_2_2n1, r_300_8_16_2_2, r_100_16_16_3_256n1, r_300_16_16, r_300_16_16n1, E_300_32_8, r_30_64_16_5_2_2_256, r_20_4_16_2_5_3_2_2048, r_300_16_16, r_300_16_16n1, E_300_32_8, r_300_16_16, r_300_16_16n1, E_2_300_16_16, E_9600_16, r_300_7_13_256, r_200_16_16_3_256, E_2_4_2_4_32_2_2_4, r_1950_28_7_2_975, r_200_16_16_3_256, E_8192_2_2, r_600_4_16_16, E_1024_2_2_2_4, E_2048_2_2_4, E_8192_2_2, E_2048_2_2_4n1, E_4096_2_4, E_2048_2_2_4, E_8192_2_2, E_1024_2_8_2, E_2048_2_8, E_4096_2_4, E_2048_2_2_4, E_8192_2_2, E_512_2_2_4_4, E_1024_2_16, E_2048_2_8, E_4096_2_4, E_2048_2_2_4, E_8192_2_2, E_64_2_2_32_4, E_64_2_32_8, E_1024_2_16, E_2048_2_8, E_4096_2_4, E_2048_2_2_4, E_8192_2_2, E_16_2_2_64_8, E_256_2_16_4, E_64_2_32_8, E_1024_2_16, E_2048_2_8, E_4096_2_4, E_2048_2_2_4, E_8192_2_2, E_64_2_128_2, E_32_2_128_4, E_256_2_16_4, E_64_2_32_8, E_1024_2_16, E_2048_2_8, E_4096_2_4, E_2048_2_2_4, E_8192_2_2, E_32_2_2_32_8, E_16_2_256_4, E_32_2_128_4, E_256_2_16_4, E_64_2_32_8, E_1024_2_16, E_2048_2_8, E_4096_2_4, E_2048_2_2_4, E_8192_2_2, E_4_2_2_512_4, E_8_2_512_4, E_16_2_256_4, E_32_2_128_4, E_256_2_16_4, E_64_2_32_8, E_1024_2_16, E_2048_2_8, E_4096_2_4, E_2048_2_2_4, E_8192_2_2, E_2_2_1024_8, E_2_1024_16, E_8_2_512_4, E_16_2_256_4, E_32_2_128_4, E_256_2_16_4, E_64_2_32_8, E_1024_2_16, E_2048_2_8, E_4096_2_4, E_2048_2_2_4, E_8192_2_2, E_4_2_2048_2, E_2_2048_8, E_2_1024_16, E_8_2_512_4, E_16_2_256_4, E_32_2_128_4, E_256_2_16_4, E_64_2_32_8, E_1024_2_16, E_2048_2_8, E_4096_2_4, E_2048_2_2_4, E_8192_2_2, E_2_2_2_1024_4, E_2_4096_4, E_2_2048_8, E_2_1024_16, E_8_2_512_4, E_16_2_256_4, E_32_2_128_4, E_256_2_16_4, E_64_2_32_8, E_1024_2_16, E_2048_2_8, E_4096_2_4, E_2048_2_2_4, E_8192_2_2, E_2_2_2048_4, E_2_2_2048_4n1, E_2_4096_4, E_2_2048_8, E_2_1024_16, E_8_2_512_4, E_16_2_256_4, E_32_2_128_4, E_256_2_16_4, E_64_2_32_8, E_1024_2_16, E_2048_2_8, E_4096_2_4, E_2048_2_2_4, E_8192_2_2, E_2_4096_4n1, E_2_2_2048_4n1, E_2_4096_4, E_2_2048_8, E_2_1024_16, E_8_2_512_4, E_16_2_256_4, E_32_2_128_4, E_256_2_16_4, E_64_2_32_8, E_1024_2_16, E_2048_2_8, E_4096_2_4, E_2048_2_2_4, E_8192_2_2, r_1950_28_7_2_975n1, r_75_28_4_975, E_300_6, E_300_6n1];
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
        addComputePass(device, commandEncoder, pipelines[1], layouts[1], infinityBuf, [buf_1, buf_2, buf_3, buf_4], [16, 100, 1]);
        addComputePass(device, commandEncoder, pipelines[2], layouts[2], infinityBuf, [input0, buf_0], [9216, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[3], layouts[3], infinityBuf, [buf_0, input0], [24, 384, 1]);
        addComputePass(device, commandEncoder, pipelines[4], layouts[4], infinityBuf, [buf_5, buf_0, buf_6, buf_7], [12, 384, 1]);
        addComputePass(device, commandEncoder, pipelines[5], layouts[5], infinityBuf, [buf_8, buf_9, buf_5, buf_10, buf_11, buf_12], [24, 577, 1]);
        addComputePass(device, commandEncoder, pipelines[6], layouts[6], infinityBuf, [buf_13, buf_8], [145, 2, 2]);
        addComputePass(device, commandEncoder, pipelines[7], layouts[7], infinityBuf, [buf_14, buf_8, buf_13], [145, 2, 2]);
        addComputePass(device, commandEncoder, pipelines[8], layouts[8], infinityBuf, [buf_15, buf_8, buf_13, buf_14, buf_16, buf_17], [24, 145, 2]);
        addComputePass(device, commandEncoder, pipelines[9], layouts[9], infinityBuf, [buf_18, buf_15, buf_19, buf_20], [24, 116, 1]);
        addComputePass(device, commandEncoder, pipelines[10], layouts[10], infinityBuf, [buf_21, buf_15, buf_22, buf_23], [24, 116, 1]);
        addComputePass(device, commandEncoder, pipelines[11], layouts[11], infinityBuf, [buf_24, buf_15, buf_25, buf_26], [24, 116, 1]);
        addComputePass(device, commandEncoder, pipelines[12], layouts[12], infinityBuf, [buf_27, buf_18, buf_21], [725, 6, 4]);
        addComputePass(device, commandEncoder, pipelines[13], layouts[13], infinityBuf, [buf_28, buf_27], [435, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[14], layouts[14], infinityBuf, [buf_29, buf_27, buf_28], [435, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[15], layouts[15], infinityBuf, [buf_21, buf_27, buf_28, buf_29, buf_24], [145, 4, 1]);
        addComputePass(device, commandEncoder, pipelines[16], layouts[16], infinityBuf, [buf_24, buf_21, buf_30, buf_31, buf_32, buf_8], [348, 2, 2]);
        addComputePass(device, commandEncoder, pipelines[17], layouts[17], infinityBuf, [buf_14, buf_24], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[18], layouts[18], infinityBuf, [buf_13, buf_24, buf_14], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[19], layouts[19], infinityBuf, [buf_21, buf_24, buf_14, buf_13, buf_33, buf_34], [24, 290, 1]);
        addComputePass(device, commandEncoder, pipelines[20], layouts[20], infinityBuf, [buf_35, buf_21, buf_36, buf_37], [32, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[21], layouts[21], infinityBuf, [buf_21, buf_35, buf_38, buf_39, buf_40, buf_24], [32, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[22], layouts[22], infinityBuf, [buf_13, buf_21], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[23], layouts[23], infinityBuf, [buf_14, buf_21, buf_13], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[24], layouts[24], infinityBuf, [buf_24, buf_21, buf_13, buf_14, buf_41, buf_42], [24, 290, 1]);
        addComputePass(device, commandEncoder, pipelines[25], layouts[25], infinityBuf, [buf_18, buf_24, buf_43, buf_44], [24, 116, 1]);
        addComputePass(device, commandEncoder, pipelines[26], layouts[26], infinityBuf, [buf_15, buf_24, buf_45, buf_46], [24, 116, 1]);
        addComputePass(device, commandEncoder, pipelines[27], layouts[27], infinityBuf, [buf_47, buf_24, buf_48, buf_49], [24, 116, 1]);
        addComputePass(device, commandEncoder, pipelines[28], layouts[28], infinityBuf, [buf_27, buf_18, buf_15], [725, 6, 4]);
        addComputePass(device, commandEncoder, pipelines[29], layouts[29], infinityBuf, [buf_29, buf_27], [435, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[30], layouts[30], infinityBuf, [buf_28, buf_27, buf_29], [435, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[31], layouts[31], infinityBuf, [buf_15, buf_27, buf_29, buf_28, buf_47], [145, 4, 1]);
        addComputePass(device, commandEncoder, pipelines[32], layouts[32], infinityBuf, [buf_47, buf_15, buf_50, buf_51, buf_52, buf_21], [32, 10, 1]);
        addComputePass(device, commandEncoder, pipelines[33], layouts[33], infinityBuf, [buf_14, buf_47], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[34], layouts[34], infinityBuf, [buf_13, buf_47, buf_14], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[35], layouts[35], infinityBuf, [buf_15, buf_47, buf_14, buf_13, buf_53, buf_54], [24, 290, 1]);
        addComputePass(device, commandEncoder, pipelines[36], layouts[36], infinityBuf, [buf_35, buf_15, buf_55, buf_56], [32, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[37], layouts[37], infinityBuf, [buf_15, buf_35, buf_57, buf_58, buf_59, buf_47], [32, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[38], layouts[38], infinityBuf, [buf_13, buf_15], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[39], layouts[39], infinityBuf, [buf_14, buf_15, buf_13], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[40], layouts[40], infinityBuf, [buf_47, buf_15, buf_13, buf_14, buf_60, buf_61], [24, 290, 1]);
        addComputePass(device, commandEncoder, pipelines[41], layouts[41], infinityBuf, [buf_21, buf_47, buf_62, buf_63], [24, 116, 1]);
        addComputePass(device, commandEncoder, pipelines[42], layouts[42], infinityBuf, [buf_18, buf_47, buf_64, buf_65], [24, 116, 1]);
        addComputePass(device, commandEncoder, pipelines[43], layouts[43], infinityBuf, [buf_24, buf_47, buf_66, buf_67], [24, 116, 1]);
        addComputePass(device, commandEncoder, pipelines[44], layouts[44], infinityBuf, [buf_27, buf_21, buf_18], [725, 6, 4]);
        addComputePass(device, commandEncoder, pipelines[45], layouts[45], infinityBuf, [buf_28, buf_27], [435, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[46], layouts[46], infinityBuf, [buf_29, buf_27, buf_28], [435, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[47], layouts[47], infinityBuf, [buf_18, buf_27, buf_28, buf_29, buf_24], [145, 4, 1]);
        addComputePass(device, commandEncoder, pipelines[48], layouts[48], infinityBuf, [buf_24, buf_18, buf_68, buf_69, buf_70, buf_15], [32, 10, 1]);
        addComputePass(device, commandEncoder, pipelines[49], layouts[49], infinityBuf, [buf_14, buf_24], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[50], layouts[50], infinityBuf, [buf_13, buf_24, buf_14], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[51], layouts[51], infinityBuf, [buf_18, buf_24, buf_14, buf_13, buf_71, buf_72], [24, 290, 1]);
        addComputePass(device, commandEncoder, pipelines[52], layouts[52], infinityBuf, [buf_35, buf_18, buf_73, buf_74], [32, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[53], layouts[53], infinityBuf, [buf_18, buf_35, buf_75, buf_76, buf_77, buf_24], [32, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[54], layouts[54], infinityBuf, [buf_13, buf_18], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[55], layouts[55], infinityBuf, [buf_14, buf_18], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[56], layouts[56], infinityBuf, [buf_78, buf_18, buf_13], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[57], layouts[57], infinityBuf, [buf_79, buf_18, buf_14], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[58], layouts[58], infinityBuf, [buf_24, buf_18, buf_13, buf_78, buf_80, buf_81], [48, 580, 1]);
        addComputePass(device, commandEncoder, pipelines[59], layouts[59], infinityBuf, [buf_82, buf_18, buf_14, buf_79, buf_83, buf_84], [24, 2, 192]);
        addComputePass(device, commandEncoder, pipelines[60], layouts[60], infinityBuf, [buf_15, buf_24, buf_85, buf_86], [32, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[61], layouts[61], infinityBuf, [buf_21, buf_24, buf_87, buf_88], [32, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[62], layouts[62], infinityBuf, [buf_47, buf_24, buf_89, buf_90], [32, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[63], layouts[63], infinityBuf, [buf_91, buf_15, buf_21], [20, 116, 6]);
        addComputePass(device, commandEncoder, pipelines[64], layouts[64], infinityBuf, [buf_29, buf_91], [120, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[65], layouts[65], infinityBuf, [buf_28, buf_91, buf_29], [120, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[66], layouts[66], infinityBuf, [buf_21, buf_91, buf_29, buf_28, buf_47], [6, 145, 1]);
        addComputePass(device, commandEncoder, pipelines[67], layouts[67], infinityBuf, [buf_47, buf_21, buf_92, buf_93, buf_94, buf_18], [32, 10, 1]);
        addComputePass(device, commandEncoder, pipelines[68], layouts[68], infinityBuf, [buf_79, buf_47], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[69], layouts[69], infinityBuf, [buf_14, buf_47, buf_79], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[70], layouts[70], infinityBuf, [buf_21, buf_47, buf_79, buf_14, buf_95, buf_96], [24, 290, 1]);
        addComputePass(device, commandEncoder, pipelines[71], layouts[71], infinityBuf, [buf_35, buf_21, buf_97, buf_98], [32, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[72], layouts[72], infinityBuf, [buf_21, buf_35, buf_99, buf_100, buf_101, buf_47], [32, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[73], layouts[73], infinityBuf, [buf_14, buf_21], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[74], layouts[74], infinityBuf, [buf_79, buf_21, buf_14], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[75], layouts[75], infinityBuf, [buf_47, buf_21, buf_14, buf_79, buf_102, buf_103], [24, 290, 1]);
        addComputePass(device, commandEncoder, pipelines[76], layouts[76], infinityBuf, [buf_18, buf_47, buf_104, buf_105], [24, 116, 1]);
        addComputePass(device, commandEncoder, pipelines[77], layouts[77], infinityBuf, [buf_15, buf_47, buf_106, buf_107], [24, 116, 1]);
        addComputePass(device, commandEncoder, pipelines[78], layouts[78], infinityBuf, [buf_24, buf_47, buf_108, buf_109], [24, 116, 1]);
        addComputePass(device, commandEncoder, pipelines[79], layouts[79], infinityBuf, [buf_27, buf_18, buf_15], [725, 6, 4]);
        addComputePass(device, commandEncoder, pipelines[80], layouts[80], infinityBuf, [buf_28, buf_27], [435, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[81], layouts[81], infinityBuf, [buf_29, buf_27, buf_28], [435, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[82], layouts[82], infinityBuf, [buf_15, buf_27, buf_28, buf_29, buf_24], [145, 4, 1]);
        addComputePass(device, commandEncoder, pipelines[83], layouts[83], infinityBuf, [buf_24, buf_15, buf_110, buf_111, buf_112, buf_21], [32, 10, 1]);
        addComputePass(device, commandEncoder, pipelines[84], layouts[84], infinityBuf, [buf_79, buf_24], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[85], layouts[85], infinityBuf, [buf_14, buf_24, buf_79], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[86], layouts[86], infinityBuf, [buf_15, buf_24, buf_79, buf_14, buf_113, buf_114], [24, 290, 1]);
        addComputePass(device, commandEncoder, pipelines[87], layouts[87], infinityBuf, [buf_35, buf_15, buf_115, buf_116], [32, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[88], layouts[88], infinityBuf, [buf_15, buf_35, buf_117, buf_118, buf_119, buf_24], [32, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[89], layouts[89], infinityBuf, [buf_14, buf_15], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[90], layouts[90], infinityBuf, [buf_79, buf_15, buf_14], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[91], layouts[91], infinityBuf, [buf_24, buf_15, buf_14, buf_79, buf_120, buf_121], [24, 290, 1]);
        addComputePass(device, commandEncoder, pipelines[92], layouts[92], infinityBuf, [buf_21, buf_24, buf_122, buf_123], [24, 116, 1]);
        addComputePass(device, commandEncoder, pipelines[93], layouts[93], infinityBuf, [buf_18, buf_24, buf_124, buf_125], [24, 116, 1]);
        addComputePass(device, commandEncoder, pipelines[94], layouts[94], infinityBuf, [buf_47, buf_24, buf_126, buf_127], [24, 116, 1]);
        addComputePass(device, commandEncoder, pipelines[95], layouts[95], infinityBuf, [buf_27, buf_21, buf_18], [725, 6, 4]);
        addComputePass(device, commandEncoder, pipelines[96], layouts[96], infinityBuf, [buf_29, buf_27], [435, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[97], layouts[97], infinityBuf, [buf_28, buf_27, buf_29], [435, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[98], layouts[98], infinityBuf, [buf_18, buf_27, buf_29, buf_28, buf_47], [145, 4, 1]);
        addComputePass(device, commandEncoder, pipelines[99], layouts[99], infinityBuf, [buf_47, buf_18, buf_128, buf_129, buf_130, buf_15], [32, 10, 1]);
        addComputePass(device, commandEncoder, pipelines[100], layouts[100], infinityBuf, [buf_79, buf_47], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[101], layouts[101], infinityBuf, [buf_14, buf_47, buf_79], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[102], layouts[102], infinityBuf, [buf_18, buf_47, buf_79, buf_14, buf_131, buf_132], [24, 290, 1]);
        addComputePass(device, commandEncoder, pipelines[103], layouts[103], infinityBuf, [buf_35, buf_18, buf_133, buf_134], [32, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[104], layouts[104], infinityBuf, [buf_18, buf_35, buf_135, buf_136, buf_137, buf_47], [32, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[105], layouts[105], infinityBuf, [buf_14, buf_18], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[106], layouts[106], infinityBuf, [buf_79, buf_18], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[107], layouts[107], infinityBuf, [buf_78, buf_18, buf_14], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[108], layouts[108], infinityBuf, [buf_13, buf_18, buf_79], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[109], layouts[109], infinityBuf, [buf_47, buf_18, buf_14, buf_78, buf_138, buf_139], [48, 580, 1]);
        addComputePass(device, commandEncoder, pipelines[110], layouts[110], infinityBuf, [buf_140, buf_18, buf_79, buf_13, buf_83, buf_84], [24, 2, 192]);
        addComputePass(device, commandEncoder, pipelines[111], layouts[111], infinityBuf, [buf_15, buf_47, buf_141, buf_142], [32, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[112], layouts[112], infinityBuf, [buf_21, buf_47, buf_143, buf_144], [32, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[113], layouts[113], infinityBuf, [buf_24, buf_47, buf_145, buf_146], [32, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[114], layouts[114], infinityBuf, [buf_91, buf_15, buf_21], [20, 116, 6]);
        addComputePass(device, commandEncoder, pipelines[115], layouts[115], infinityBuf, [buf_28, buf_91], [120, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[116], layouts[116], infinityBuf, [buf_29, buf_91, buf_28], [120, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[117], layouts[117], infinityBuf, [buf_21, buf_91, buf_28, buf_29, buf_24], [6, 145, 1]);
        addComputePass(device, commandEncoder, pipelines[118], layouts[118], infinityBuf, [buf_24, buf_21, buf_147, buf_148, buf_149, buf_18], [32, 10, 1]);
        addComputePass(device, commandEncoder, pipelines[119], layouts[119], infinityBuf, [buf_13, buf_24], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[120], layouts[120], infinityBuf, [buf_79, buf_24, buf_13], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[121], layouts[121], infinityBuf, [buf_21, buf_24, buf_13, buf_79, buf_150, buf_151], [24, 290, 1]);
        addComputePass(device, commandEncoder, pipelines[122], layouts[122], infinityBuf, [buf_35, buf_21, buf_152, buf_153], [32, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[123], layouts[123], infinityBuf, [buf_21, buf_35, buf_154, buf_155, buf_156, buf_24], [32, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[124], layouts[124], infinityBuf, [buf_79, buf_21], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[125], layouts[125], infinityBuf, [buf_13, buf_21, buf_79], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[126], layouts[126], infinityBuf, [buf_24, buf_21, buf_79, buf_13, buf_157, buf_158], [24, 290, 1]);
        addComputePass(device, commandEncoder, pipelines[127], layouts[127], infinityBuf, [buf_18, buf_24, buf_159, buf_160], [24, 116, 1]);
        addComputePass(device, commandEncoder, pipelines[128], layouts[128], infinityBuf, [buf_15, buf_24, buf_161, buf_162], [24, 116, 1]);
        addComputePass(device, commandEncoder, pipelines[129], layouts[129], infinityBuf, [buf_47, buf_24, buf_163, buf_164], [24, 116, 1]);
        addComputePass(device, commandEncoder, pipelines[130], layouts[130], infinityBuf, [buf_27, buf_18, buf_15], [725, 6, 4]);
        addComputePass(device, commandEncoder, pipelines[131], layouts[131], infinityBuf, [buf_29, buf_27], [435, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[132], layouts[132], infinityBuf, [buf_28, buf_27, buf_29], [435, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[133], layouts[133], infinityBuf, [buf_15, buf_27, buf_29, buf_28, buf_47], [145, 4, 1]);
        addComputePass(device, commandEncoder, pipelines[134], layouts[134], infinityBuf, [buf_47, buf_15, buf_165, buf_166, buf_167, buf_21], [32, 10, 1]);
        addComputePass(device, commandEncoder, pipelines[135], layouts[135], infinityBuf, [buf_13, buf_47], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[136], layouts[136], infinityBuf, [buf_79, buf_47, buf_13], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[137], layouts[137], infinityBuf, [buf_15, buf_47, buf_13, buf_79, buf_168, buf_169], [24, 290, 1]);
        addComputePass(device, commandEncoder, pipelines[138], layouts[138], infinityBuf, [buf_35, buf_15, buf_170, buf_171], [32, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[139], layouts[139], infinityBuf, [buf_15, buf_35, buf_172, buf_173, buf_174, buf_47], [32, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[140], layouts[140], infinityBuf, [buf_79, buf_15], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[141], layouts[141], infinityBuf, [buf_13, buf_15, buf_79], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[142], layouts[142], infinityBuf, [buf_47, buf_15, buf_79, buf_13, buf_175, buf_176], [24, 290, 1]);
        addComputePass(device, commandEncoder, pipelines[143], layouts[143], infinityBuf, [buf_21, buf_47, buf_177, buf_178], [24, 116, 1]);
        addComputePass(device, commandEncoder, pipelines[144], layouts[144], infinityBuf, [buf_18, buf_47, buf_179, buf_180], [24, 116, 1]);
        addComputePass(device, commandEncoder, pipelines[145], layouts[145], infinityBuf, [buf_24, buf_47, buf_181, buf_182], [24, 116, 1]);
        addComputePass(device, commandEncoder, pipelines[146], layouts[146], infinityBuf, [buf_27, buf_21, buf_18], [725, 6, 4]);
        addComputePass(device, commandEncoder, pipelines[147], layouts[147], infinityBuf, [buf_28, buf_27], [435, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[148], layouts[148], infinityBuf, [buf_29, buf_27, buf_28], [435, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[149], layouts[149], infinityBuf, [buf_18, buf_27, buf_28, buf_29, buf_24], [145, 4, 1]);
        addComputePass(device, commandEncoder, pipelines[150], layouts[150], infinityBuf, [buf_24, buf_18, buf_183, buf_184, buf_185, buf_15], [32, 10, 1]);
        addComputePass(device, commandEncoder, pipelines[151], layouts[151], infinityBuf, [buf_13, buf_24], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[152], layouts[152], infinityBuf, [buf_79, buf_24, buf_13], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[153], layouts[153], infinityBuf, [buf_18, buf_24, buf_13, buf_79, buf_186, buf_187], [24, 290, 1]);
        addComputePass(device, commandEncoder, pipelines[154], layouts[154], infinityBuf, [buf_35, buf_18, buf_188, buf_189], [32, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[155], layouts[155], infinityBuf, [buf_18, buf_35, buf_190, buf_191, buf_192, buf_24], [32, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[156], layouts[156], infinityBuf, [buf_79, buf_18], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[157], layouts[157], infinityBuf, [buf_13, buf_18], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[158], layouts[158], infinityBuf, [buf_78, buf_18, buf_79], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[159], layouts[159], infinityBuf, [buf_14, buf_18, buf_13], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[160], layouts[160], infinityBuf, [buf_24, buf_18, buf_79, buf_78, buf_193, buf_194], [48, 580, 1]);
        addComputePass(device, commandEncoder, pipelines[161], layouts[161], infinityBuf, [buf_195, buf_18, buf_13, buf_14, buf_83, buf_84], [24, 2, 192]);
        addComputePass(device, commandEncoder, pipelines[162], layouts[162], infinityBuf, [buf_15, buf_24, buf_196, buf_197], [32, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[163], layouts[163], infinityBuf, [buf_21, buf_24, buf_198, buf_199], [32, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[164], layouts[164], infinityBuf, [buf_47, buf_24, buf_200, buf_201], [32, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[165], layouts[165], infinityBuf, [buf_91, buf_15, buf_21], [20, 116, 6]);
        addComputePass(device, commandEncoder, pipelines[166], layouts[166], infinityBuf, [buf_29, buf_91], [120, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[167], layouts[167], infinityBuf, [buf_28, buf_91, buf_29], [120, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[168], layouts[168], infinityBuf, [buf_21, buf_91, buf_29, buf_28, buf_47], [6, 145, 1]);
        addComputePass(device, commandEncoder, pipelines[169], layouts[169], infinityBuf, [buf_47, buf_21, buf_202, buf_203, buf_204, buf_18], [32, 10, 1]);
        addComputePass(device, commandEncoder, pipelines[170], layouts[170], infinityBuf, [buf_14, buf_47], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[171], layouts[171], infinityBuf, [buf_13, buf_47, buf_14], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[172], layouts[172], infinityBuf, [buf_21, buf_47, buf_14, buf_13, buf_205, buf_206], [24, 290, 1]);
        addComputePass(device, commandEncoder, pipelines[173], layouts[173], infinityBuf, [buf_35, buf_21, buf_207, buf_208], [32, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[174], layouts[174], infinityBuf, [buf_21, buf_35, buf_209, buf_210, buf_211, buf_47], [32, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[175], layouts[175], infinityBuf, [buf_13, buf_21], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[176], layouts[176], infinityBuf, [buf_14, buf_21, buf_13], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[177], layouts[177], infinityBuf, [buf_47, buf_21, buf_13, buf_14, buf_212, buf_213], [24, 290, 1]);
        addComputePass(device, commandEncoder, pipelines[178], layouts[178], infinityBuf, [buf_18, buf_47, buf_214, buf_215], [24, 116, 1]);
        addComputePass(device, commandEncoder, pipelines[179], layouts[179], infinityBuf, [buf_15, buf_47, buf_216, buf_217], [24, 116, 1]);
        addComputePass(device, commandEncoder, pipelines[180], layouts[180], infinityBuf, [buf_24, buf_47, buf_218, buf_219], [24, 116, 1]);
        addComputePass(device, commandEncoder, pipelines[181], layouts[181], infinityBuf, [buf_27, buf_18, buf_15], [725, 6, 4]);
        addComputePass(device, commandEncoder, pipelines[182], layouts[182], infinityBuf, [buf_28, buf_27], [435, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[183], layouts[183], infinityBuf, [buf_29, buf_27, buf_28], [435, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[184], layouts[184], infinityBuf, [buf_15, buf_27, buf_28, buf_29, buf_24], [145, 4, 1]);
        addComputePass(device, commandEncoder, pipelines[185], layouts[185], infinityBuf, [buf_24, buf_15, buf_220, buf_221, buf_222, buf_21], [32, 10, 1]);
        addComputePass(device, commandEncoder, pipelines[186], layouts[186], infinityBuf, [buf_14, buf_24], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[187], layouts[187], infinityBuf, [buf_13, buf_24, buf_14], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[188], layouts[188], infinityBuf, [buf_15, buf_24, buf_14, buf_13, buf_223, buf_224], [24, 290, 1]);
        addComputePass(device, commandEncoder, pipelines[189], layouts[189], infinityBuf, [buf_35, buf_15, buf_225, buf_226], [32, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[190], layouts[190], infinityBuf, [buf_15, buf_35, buf_227, buf_228, buf_229, buf_24], [32, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[191], layouts[191], infinityBuf, [buf_13, buf_15], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[192], layouts[192], infinityBuf, [buf_14, buf_15, buf_13], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[193], layouts[193], infinityBuf, [buf_24, buf_15, buf_13, buf_14, buf_230, buf_231], [24, 290, 1]);
        addComputePass(device, commandEncoder, pipelines[194], layouts[194], infinityBuf, [buf_21, buf_24, buf_232, buf_233], [24, 116, 1]);
        addComputePass(device, commandEncoder, pipelines[195], layouts[195], infinityBuf, [buf_18, buf_24, buf_234, buf_235], [24, 116, 1]);
        addComputePass(device, commandEncoder, pipelines[196], layouts[196], infinityBuf, [buf_47, buf_24, buf_236, buf_237], [24, 116, 1]);
        addComputePass(device, commandEncoder, pipelines[197], layouts[197], infinityBuf, [buf_27, buf_21, buf_18], [725, 6, 4]);
        addComputePass(device, commandEncoder, pipelines[198], layouts[198], infinityBuf, [buf_29, buf_27], [435, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[199], layouts[199], infinityBuf, [buf_28, buf_27, buf_29], [435, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[200], layouts[200], infinityBuf, [buf_18, buf_27, buf_29, buf_28, buf_47], [145, 4, 1]);
        addComputePass(device, commandEncoder, pipelines[201], layouts[201], infinityBuf, [buf_47, buf_18, buf_238, buf_239, buf_240, buf_15], [32, 10, 1]);
        addComputePass(device, commandEncoder, pipelines[202], layouts[202], infinityBuf, [buf_14, buf_47], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[203], layouts[203], infinityBuf, [buf_13, buf_47, buf_14], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[204], layouts[204], infinityBuf, [buf_18, buf_47, buf_14, buf_13, buf_241, buf_242], [24, 290, 1]);
        addComputePass(device, commandEncoder, pipelines[205], layouts[205], infinityBuf, [buf_35, buf_18, buf_243, buf_244], [32, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[206], layouts[206], infinityBuf, [buf_18, buf_35, buf_245, buf_246, buf_247, buf_47], [32, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[207], layouts[207], infinityBuf, [buf_13, buf_18], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[208], layouts[208], infinityBuf, [buf_14, buf_18, buf_13], [580, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[209], layouts[209], infinityBuf, [buf_248, buf_18, buf_13, buf_14, buf_83, buf_84], [24, 2, 192]);
        addComputePass(device, commandEncoder, pipelines[210], layouts[210], infinityBuf, [buf_249, buf_82, buf_140, buf_195, buf_248], [13824, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[211], layouts[211], infinityBuf, [buf_250, buf_249, buf_251], [64, 18, 1]);
        addComputePass(device, commandEncoder, pipelines[212], layouts[212], infinityBuf, [buf_252, buf_250], [144, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[213], layouts[213], infinityBuf, [buf_253, buf_250, buf_252], [32, 192, 1]);
        addComputePass(device, commandEncoder, pipelines[214], layouts[214], infinityBuf, [buf_252, buf_253], [576, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[215], layouts[215], infinityBuf, [buf_250, buf_253, buf_252, buf_254, buf_255], [24, 256, 1]);
        addComputePass(device, commandEncoder, pipelines[216], layouts[216], infinityBuf, [buf_256, buf_250, buf_257], [6, 6, 1]);
        addComputePass(device, commandEncoder, pipelines[217], layouts[217], infinityBuf, [buf_252, buf_256], [576, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[218], layouts[218], infinityBuf, [buf_258, buf_256, buf_252], [128, 36, 1]);
        addComputePass(device, commandEncoder, pipelines[219], layouts[219], infinityBuf, [buf_252, buf_258], [576, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[220], layouts[220], infinityBuf, [buf_256, buf_258, buf_252, buf_259, buf_260], [576, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[221], layouts[221], infinityBuf, [buf_258, buf_256, buf_261], [8, 12, 1]);
        addComputePass(device, commandEncoder, pipelines[222], layouts[222], infinityBuf, [buf_252, buf_258], [576, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[223], layouts[223], infinityBuf, [buf_256, buf_258, buf_252], [128, 36, 1]);
        addComputePass(device, commandEncoder, pipelines[224], layouts[224], infinityBuf, [buf_252, buf_256], [576, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[225], layouts[225], infinityBuf, [buf_258, buf_256, buf_252, buf_262, buf_263], [576, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[226], layouts[226], infinityBuf, [buf_256, buf_258, buf_264], [8, 12, 1]);
        addComputePass(device, commandEncoder, pipelines[227], layouts[227], infinityBuf, [buf_252, buf_256], [576, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[228], layouts[228], infinityBuf, [buf_265, buf_256, buf_252], [128, 36, 1]);
        addComputePass(device, commandEncoder, pipelines[229], layouts[229], infinityBuf, [buf_252, buf_265], [576, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[230], layouts[230], infinityBuf, [buf_256, buf_265, buf_252, buf_266, buf_267], [576, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[231], layouts[231], infinityBuf, [buf_265, buf_256, buf_268], [8, 12, 1]);
        addComputePass(device, commandEncoder, pipelines[232], layouts[232], infinityBuf, [buf_252, buf_265], [576, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[233], layouts[233], infinityBuf, [buf_256, buf_265, buf_252], [128, 36, 1]);
        addComputePass(device, commandEncoder, pipelines[234], layouts[234], infinityBuf, [buf_252, buf_256], [576, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[235], layouts[235], infinityBuf, [buf_265, buf_256, buf_252, buf_269, buf_270], [576, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[236], layouts[236], infinityBuf, [buf_256, buf_265, buf_271], [8, 12, 1]);
        addComputePass(device, commandEncoder, pipelines[237], layouts[237], infinityBuf, [buf_252, buf_256], [576, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[238], layouts[238], infinityBuf, [buf_272, buf_256, buf_252], [128, 36, 1]);
        addComputePass(device, commandEncoder, pipelines[239], layouts[239], infinityBuf, [buf_252, buf_272], [576, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[240], layouts[240], infinityBuf, [buf_256, buf_272, buf_252, buf_273, buf_274], [576, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[241], layouts[241], infinityBuf, [buf_272, buf_256, buf_275], [8, 12, 1]);
        addComputePass(device, commandEncoder, pipelines[242], layouts[242], infinityBuf, [buf_252, buf_272], [576, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[243], layouts[243], infinityBuf, [buf_256, buf_272, buf_252], [128, 36, 1]);
        addComputePass(device, commandEncoder, pipelines[244], layouts[244], infinityBuf, [buf_252, buf_256], [576, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[245], layouts[245], infinityBuf, [buf_276, buf_250, buf_258, buf_265, buf_256, buf_252, buf_277, buf_278], [144, 40, 1]);
        addComputePass(device, commandEncoder, pipelines[246], layouts[246], infinityBuf, [buf_250, buf_276, buf_279], [256, 18, 1]);
        addComputePass(device, commandEncoder, pipelines[247], layouts[247], infinityBuf, [buf_252, buf_250], [144, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[248], layouts[248], infinityBuf, [buf_253, buf_250, buf_252], [32, 192, 1]);
        addComputePass(device, commandEncoder, pipelines[249], layouts[249], infinityBuf, [buf_252, buf_253], [576, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[250], layouts[250], infinityBuf, [buf_250, buf_253, buf_252, buf_280, buf_281], [16, 576, 1]);
        addComputePass(device, commandEncoder, pipelines[251], layouts[251], infinityBuf, [buf_252, buf_250], [144, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[252], layouts[252], infinityBuf, [buf_253, buf_250, buf_252], [32, 192, 1]);
        addComputePass(device, commandEncoder, pipelines[253], layouts[253], infinityBuf, [buf_252, buf_253], [576, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[254], layouts[254], infinityBuf, [buf_250, buf_253, buf_252, buf_282, buf_283], [64, 72, 1]);
        addComputePass(device, commandEncoder, pipelines[255], layouts[255], infinityBuf, [buf_253, buf_250, buf_284, buf_285], [24, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[256], layouts[256], infinityBuf, [buf_286, buf_250, buf_287, buf_288], [16, 12, 1]);
        addComputePass(device, commandEncoder, pipelines[257], layouts[257], infinityBuf, [buf_289, buf_250, buf_290, buf_291], [24, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[258], layouts[258], infinityBuf, [buf_252, buf_286], [144, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[259], layouts[259], infinityBuf, [buf_292, buf_286, buf_252], [576, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[260], layouts[260], infinityBuf, [buf_250, buf_286, buf_252, buf_292, buf_293, buf_294], [64, 72, 1]);
        addComputePass(device, commandEncoder, pipelines[261], layouts[261], infinityBuf, [buf_292, buf_250, buf_295, buf_296], [96, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[262], layouts[262], infinityBuf, [buf_286, buf_250, buf_297, buf_298], [64, 18, 1]);
        addComputePass(device, commandEncoder, pipelines[263], layouts[263], infinityBuf, [buf_299, buf_292], [2, 256, 1]);
        addComputePass(device, commandEncoder, pipelines[264], layouts[264], infinityBuf, [buf_300, buf_292], [576, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[265], layouts[265], infinityBuf, [buf_250, buf_286, buf_301, buf_302], [64, 18, 1]);
        addComputePass(device, commandEncoder, pipelines[266], layouts[266], infinityBuf, [buf_303, buf_299], [512, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[267], layouts[267], infinityBuf, [buf_304, buf_250, buf_305, buf_306], [4, 36, 1]);
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
        addComputePass(device, commandEncoder, pipelines[330], layouts[330], infinityBuf, [buf_307, buf_303], [576, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[331], layouts[331], infinityBuf, [buf_308, buf_292, buf_303, buf_300, buf_307], [576, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[332], layouts[332], infinityBuf, [buf_309, buf_308, buf_304], [4, 75, 1]);
        addComputePass(device, commandEncoder, pipelines[333], layouts[333], infinityBuf, [buf_310, buf_311, buf_309, buf_312, buf_313], [64, 60, 1]);
        addComputePass(device, commandEncoder, pipelines[334], layouts[334], infinityBuf, [buf_314, buf_310, buf_315, buf_316], [8, 100, 1]);
        addComputePass(device, commandEncoder, pipelines[335], layouts[335], infinityBuf, [buf_310, buf_2, buf_314, buf_3, buf_4], [8, 30, 1]);
        addComputePass(device, commandEncoder, pipelines[336], layouts[336], infinityBuf, [buf_317, buf_2, buf_314, buf_3, buf_4], [16, 75, 1]);
        addComputePass(device, commandEncoder, pipelines[337], layouts[337], infinityBuf, [buf_318, buf_310, buf_317], [100, 300, 1]);
        addComputePass(device, commandEncoder, pipelines[338], layouts[338], infinityBuf, [buf_319, buf_318], [300, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[339], layouts[339], infinityBuf, [buf_320, buf_318, buf_319], [150, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[340], layouts[340], infinityBuf, [buf_317, buf_318, buf_319, buf_320, buf_1], [100, 8, 2]);
        addComputePass(device, commandEncoder, pipelines[341], layouts[341], infinityBuf, [buf_1, buf_2, buf_317, buf_321, buf_322], [2, 60, 1]);
        addComputePass(device, commandEncoder, pipelines[342], layouts[342], infinityBuf, [buf_323, buf_1], [300, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[343], layouts[343], infinityBuf, [buf_324, buf_1, buf_323], [300, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[344], layouts[344], infinityBuf, [buf_317, buf_1, buf_323, buf_324, buf_325, buf_326], [32, 300, 1]);
        addComputePass(device, commandEncoder, pipelines[345], layouts[345], infinityBuf, [buf_327, buf_311, buf_309, buf_317, buf_314, buf_328, buf_329], [2, 2, 300]);
        addComputePass(device, commandEncoder, pipelines[346], layouts[346], infinityBuf, [buf_330, buf_317, buf_314, buf_331, buf_332], [2, 300, 1]);
        addComputePass(device, commandEncoder, pipelines[347], layouts[347], infinityBuf, [buf_333, buf_330], [2400, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[348], layouts[348], infinityBuf, [buf_334, buf_330, buf_333], [2400, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[349], layouts[349], infinityBuf, [buf_1, buf_327, buf_253, buf_330, buf_333, buf_334], [8, 300, 1]);
        addComputePass(device, commandEncoder, pipelines[350], layouts[350], infinityBuf, [buf_310, buf_317, buf_1, buf_335, buf_336], [16, 100, 1]);
        addComputePass(device, commandEncoder, pipelines[351], layouts[351], infinityBuf, [buf_324, buf_310], [300, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[352], layouts[352], infinityBuf, [buf_323, buf_310, buf_324], [300, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[353], layouts[353], infinityBuf, [buf_1, buf_310, buf_324, buf_323, buf_337, buf_338], [32, 300, 1]);
        addComputePass(device, commandEncoder, pipelines[354], layouts[354], infinityBuf, [buf_339, buf_1, buf_340, buf_341], [64, 30, 1]);
        addComputePass(device, commandEncoder, pipelines[355], layouts[355], infinityBuf, [buf_310, buf_1, buf_339, buf_342, buf_343], [4, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[356], layouts[356], infinityBuf, [buf_323, buf_310], [300, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[357], layouts[357], infinityBuf, [buf_324, buf_310, buf_323], [300, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[358], layouts[358], infinityBuf, [buf_1, buf_310, buf_323, buf_324, buf_344, buf_345], [32, 300, 1]);
        addComputePass(device, commandEncoder, pipelines[359], layouts[359], infinityBuf, [buf_310, buf_1, buf_314, buf_346, buf_347], [100, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[360], layouts[360], infinityBuf, [buf_317, buf_1, buf_314, buf_346, buf_347], [100, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[361], layouts[361], infinityBuf, [buf_348, buf_1, buf_346, buf_347], [16, 75, 1]);
        addComputePass(device, commandEncoder, pipelines[362], layouts[362], infinityBuf, [buf_324, buf_1], [300, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[363], layouts[363], infinityBuf, [buf_318, buf_310, buf_317], [100, 300, 1]);
        addComputePass(device, commandEncoder, pipelines[364], layouts[364], infinityBuf, [buf_323, buf_1, buf_324], [300, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[365], layouts[365], infinityBuf, [buf_320, buf_318], [300, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[366], layouts[366], infinityBuf, [buf_349, buf_1, buf_324, buf_323, buf_350, buf_351], [8, 300, 2]);
        addComputePass(device, commandEncoder, pipelines[367], layouts[367], infinityBuf, [buf_319, buf_318, buf_320], [150, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[368], layouts[368], infinityBuf, [buf_317, buf_318, buf_320, buf_319, buf_348], [100, 8, 2]);
        addComputePass(device, commandEncoder, pipelines[369], layouts[369], infinityBuf, [buf_348, buf_1, buf_317, buf_352, buf_353], [16, 100, 1]);
        addComputePass(device, commandEncoder, pipelines[370], layouts[370], infinityBuf, [buf_323, buf_348], [300, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[371], layouts[371], infinityBuf, [buf_324, buf_348, buf_323], [300, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[372], layouts[372], infinityBuf, [buf_317, buf_348, buf_323, buf_324, buf_354, buf_355], [32, 300, 1]);
        addComputePass(device, commandEncoder, pipelines[373], layouts[373], infinityBuf, [buf_327, buf_311, buf_309, buf_317, buf_314, buf_356, buf_357], [2, 2, 300]);
        addComputePass(device, commandEncoder, pipelines[374], layouts[374], infinityBuf, [buf_330, buf_317, buf_314, buf_358, buf_359], [2, 300, 1]);
        addComputePass(device, commandEncoder, pipelines[375], layouts[375], infinityBuf, [buf_334, buf_330], [2400, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[376], layouts[376], infinityBuf, [buf_333, buf_330, buf_334], [2400, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[377], layouts[377], infinityBuf, [buf_314, buf_327, buf_289, buf_330, buf_334, buf_333], [8, 300, 1]);
        addComputePass(device, commandEncoder, pipelines[378], layouts[378], infinityBuf, [buf_348, buf_317, buf_314, buf_360, buf_361], [16, 100, 1]);
        addComputePass(device, commandEncoder, pipelines[379], layouts[379], infinityBuf, [buf_324, buf_348], [300, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[380], layouts[380], infinityBuf, [buf_323, buf_348, buf_324], [300, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[381], layouts[381], infinityBuf, [buf_314, buf_348, buf_324, buf_323, buf_362, buf_363], [32, 300, 1]);
        addComputePass(device, commandEncoder, pipelines[382], layouts[382], infinityBuf, [buf_339, buf_314, buf_364, buf_365], [64, 30, 1]);
        addComputePass(device, commandEncoder, pipelines[383], layouts[383], infinityBuf, [buf_348, buf_314, buf_339, buf_366, buf_367], [4, 20, 1]);
        addComputePass(device, commandEncoder, pipelines[384], layouts[384], infinityBuf, [buf_323, buf_348], [300, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[385], layouts[385], infinityBuf, [buf_324, buf_348, buf_323], [300, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[386], layouts[386], infinityBuf, [buf_314, buf_348, buf_323, buf_324, buf_368, buf_369], [32, 300, 1]);
        addComputePass(device, commandEncoder, pipelines[387], layouts[387], infinityBuf, [buf_324, buf_314], [300, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[388], layouts[388], infinityBuf, [buf_323, buf_314, buf_324], [300, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[389], layouts[389], infinityBuf, [buf_370, buf_314, buf_324, buf_323, buf_350, buf_351], [16, 300, 2]);
        addComputePass(device, commandEncoder, pipelines[390], layouts[390], infinityBuf, [buf_371, buf_349, buf_370], [9600, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[391], layouts[391], infinityBuf, [buf_372, buf_371, buf_373, buf_374], [7, 300, 1]);
        addComputePass(device, commandEncoder, pipelines[392], layouts[392], infinityBuf, [buf_370, buf_371, buf_375, buf_376], [16, 200, 1]);
        addComputePass(device, commandEncoder, pipelines[393], layouts[393], infinityBuf, [buf_377, buf_372], [1024, 4, 2]);
        addComputePass(device, commandEncoder, pipelines[394], layouts[394], infinityBuf, [buf_378, buf_372], [1950, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[395], layouts[395], infinityBuf, [buf_371, buf_370, buf_379, buf_380], [16, 200, 1]);
        addComputePass(device, commandEncoder, pipelines[396], layouts[396], infinityBuf, [buf_381, buf_377], [2, 8192, 1]);
        addComputePass(device, commandEncoder, pipelines[397], layouts[397], infinityBuf, [buf_319, buf_371, buf_382, buf_383], [4, 600, 1]);
        addComputePass(device, commandEncoder, pipelines[398], layouts[398], infinityBuf, [buf_377, buf_381], [4, 2, 1024]);
        addComputePass(device, commandEncoder, pipelines[399], layouts[399], infinityBuf, [buf_381, buf_377], [2, 2, 2048]);
        addComputePass(device, commandEncoder, pipelines[400], layouts[400], infinityBuf, [buf_377, buf_381], [2, 8192, 1]);
        addComputePass(device, commandEncoder, pipelines[401], layouts[401], infinityBuf, [buf_381, buf_377], [2, 2, 2048]);
        addComputePass(device, commandEncoder, pipelines[402], layouts[402], infinityBuf, [buf_377, buf_381], [2, 4096, 1]);
        addComputePass(device, commandEncoder, pipelines[403], layouts[403], infinityBuf, [buf_381, buf_377], [2, 2, 2048]);
        addComputePass(device, commandEncoder, pipelines[404], layouts[404], infinityBuf, [buf_377, buf_381], [2, 8192, 1]);
        addComputePass(device, commandEncoder, pipelines[405], layouts[405], infinityBuf, [buf_381, buf_377], [8, 2, 1024]);
        addComputePass(device, commandEncoder, pipelines[406], layouts[406], infinityBuf, [buf_377, buf_381], [2, 2048, 1]);
        addComputePass(device, commandEncoder, pipelines[407], layouts[407], infinityBuf, [buf_381, buf_377], [2, 4096, 1]);
        addComputePass(device, commandEncoder, pipelines[408], layouts[408], infinityBuf, [buf_377, buf_381], [2, 2, 2048]);
        addComputePass(device, commandEncoder, pipelines[409], layouts[409], infinityBuf, [buf_381, buf_377], [2, 8192, 1]);
        addComputePass(device, commandEncoder, pipelines[410], layouts[410], infinityBuf, [buf_377, buf_381], [8, 2, 512]);
        addComputePass(device, commandEncoder, pipelines[411], layouts[411], infinityBuf, [buf_381, buf_377], [2, 1024, 1]);
        addComputePass(device, commandEncoder, pipelines[412], layouts[412], infinityBuf, [buf_377, buf_381], [2, 2048, 1]);
        addComputePass(device, commandEncoder, pipelines[413], layouts[413], infinityBuf, [buf_381, buf_377], [2, 4096, 1]);
        addComputePass(device, commandEncoder, pipelines[414], layouts[414], infinityBuf, [buf_377, buf_381], [2, 2, 2048]);
        addComputePass(device, commandEncoder, pipelines[415], layouts[415], infinityBuf, [buf_381, buf_377], [2, 8192, 1]);
        addComputePass(device, commandEncoder, pipelines[416], layouts[416], infinityBuf, [buf_377, buf_381], [64, 2, 64]);
        addComputePass(device, commandEncoder, pipelines[417], layouts[417], infinityBuf, [buf_381, buf_377], [32, 2, 64]);
        addComputePass(device, commandEncoder, pipelines[418], layouts[418], infinityBuf, [buf_377, buf_381], [2, 1024, 1]);
        addComputePass(device, commandEncoder, pipelines[419], layouts[419], infinityBuf, [buf_381, buf_377], [2, 2048, 1]);
        addComputePass(device, commandEncoder, pipelines[420], layouts[420], infinityBuf, [buf_377, buf_381], [2, 4096, 1]);
        addComputePass(device, commandEncoder, pipelines[421], layouts[421], infinityBuf, [buf_381, buf_377], [2, 2, 2048]);
        addComputePass(device, commandEncoder, pipelines[422], layouts[422], infinityBuf, [buf_377, buf_381], [2, 8192, 1]);
        addComputePass(device, commandEncoder, pipelines[423], layouts[423], infinityBuf, [buf_381, buf_377], [128, 2, 16]);
        addComputePass(device, commandEncoder, pipelines[424], layouts[424], infinityBuf, [buf_377, buf_381], [16, 2, 256]);
        addComputePass(device, commandEncoder, pipelines[425], layouts[425], infinityBuf, [buf_381, buf_377], [32, 2, 64]);
        addComputePass(device, commandEncoder, pipelines[426], layouts[426], infinityBuf, [buf_377, buf_381], [2, 1024, 1]);
        addComputePass(device, commandEncoder, pipelines[427], layouts[427], infinityBuf, [buf_381, buf_377], [2, 2048, 1]);
        addComputePass(device, commandEncoder, pipelines[428], layouts[428], infinityBuf, [buf_377, buf_381], [2, 4096, 1]);
        addComputePass(device, commandEncoder, pipelines[429], layouts[429], infinityBuf, [buf_381, buf_377], [2, 2, 2048]);
        addComputePass(device, commandEncoder, pipelines[430], layouts[430], infinityBuf, [buf_377, buf_381], [2, 8192, 1]);
        addComputePass(device, commandEncoder, pipelines[431], layouts[431], infinityBuf, [buf_381, buf_377], [128, 2, 64]);
        addComputePass(device, commandEncoder, pipelines[432], layouts[432], infinityBuf, [buf_377, buf_381], [128, 2, 32]);
        addComputePass(device, commandEncoder, pipelines[433], layouts[433], infinityBuf, [buf_381, buf_377], [16, 2, 256]);
        addComputePass(device, commandEncoder, pipelines[434], layouts[434], infinityBuf, [buf_377, buf_381], [32, 2, 64]);
        addComputePass(device, commandEncoder, pipelines[435], layouts[435], infinityBuf, [buf_381, buf_377], [2, 1024, 1]);
        addComputePass(device, commandEncoder, pipelines[436], layouts[436], infinityBuf, [buf_377, buf_381], [2, 2048, 1]);
        addComputePass(device, commandEncoder, pipelines[437], layouts[437], infinityBuf, [buf_381, buf_377], [2, 4096, 1]);
        addComputePass(device, commandEncoder, pipelines[438], layouts[438], infinityBuf, [buf_377, buf_381], [2, 2, 2048]);
        addComputePass(device, commandEncoder, pipelines[439], layouts[439], infinityBuf, [buf_381, buf_377], [2, 8192, 1]);
        addComputePass(device, commandEncoder, pipelines[440], layouts[440], infinityBuf, [buf_377, buf_381], [64, 2, 32]);
        addComputePass(device, commandEncoder, pipelines[441], layouts[441], infinityBuf, [buf_381, buf_377], [256, 2, 16]);
        addComputePass(device, commandEncoder, pipelines[442], layouts[442], infinityBuf, [buf_377, buf_381], [128, 2, 32]);
        addComputePass(device, commandEncoder, pipelines[443], layouts[443], infinityBuf, [buf_381, buf_377], [16, 2, 256]);
        addComputePass(device, commandEncoder, pipelines[444], layouts[444], infinityBuf, [buf_377, buf_381], [32, 2, 64]);
        addComputePass(device, commandEncoder, pipelines[445], layouts[445], infinityBuf, [buf_381, buf_377], [2, 1024, 1]);
        addComputePass(device, commandEncoder, pipelines[446], layouts[446], infinityBuf, [buf_377, buf_381], [2, 2048, 1]);
        addComputePass(device, commandEncoder, pipelines[447], layouts[447], infinityBuf, [buf_381, buf_377], [2, 4096, 1]);
        addComputePass(device, commandEncoder, pipelines[448], layouts[448], infinityBuf, [buf_377, buf_381], [2, 2, 2048]);
        addComputePass(device, commandEncoder, pipelines[449], layouts[449], infinityBuf, [buf_381, buf_377], [2, 8192, 1]);
        addComputePass(device, commandEncoder, pipelines[450], layouts[450], infinityBuf, [buf_377, buf_381], [1024, 2, 4]);
        addComputePass(device, commandEncoder, pipelines[451], layouts[451], infinityBuf, [buf_381, buf_377], [512, 2, 8]);
        addComputePass(device, commandEncoder, pipelines[452], layouts[452], infinityBuf, [buf_377, buf_381], [256, 2, 16]);
        addComputePass(device, commandEncoder, pipelines[453], layouts[453], infinityBuf, [buf_381, buf_377], [128, 2, 32]);
        addComputePass(device, commandEncoder, pipelines[454], layouts[454], infinityBuf, [buf_377, buf_381], [16, 2, 256]);
        addComputePass(device, commandEncoder, pipelines[455], layouts[455], infinityBuf, [buf_381, buf_377], [32, 2, 64]);
        addComputePass(device, commandEncoder, pipelines[456], layouts[456], infinityBuf, [buf_377, buf_381], [2, 1024, 1]);
        addComputePass(device, commandEncoder, pipelines[457], layouts[457], infinityBuf, [buf_381, buf_377], [2, 2048, 1]);
        addComputePass(device, commandEncoder, pipelines[458], layouts[458], infinityBuf, [buf_377, buf_381], [2, 4096, 1]);
        addComputePass(device, commandEncoder, pipelines[459], layouts[459], infinityBuf, [buf_381, buf_377], [2, 2, 2048]);
        addComputePass(device, commandEncoder, pipelines[460], layouts[460], infinityBuf, [buf_377, buf_381], [2, 8192, 1]);
        addComputePass(device, commandEncoder, pipelines[461], layouts[461], infinityBuf, [buf_381, buf_377], [1024, 2, 2]);
        addComputePass(device, commandEncoder, pipelines[462], layouts[462], infinityBuf, [buf_377, buf_381], [1024, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[463], layouts[463], infinityBuf, [buf_381, buf_377], [512, 2, 8]);
        addComputePass(device, commandEncoder, pipelines[464], layouts[464], infinityBuf, [buf_377, buf_381], [256, 2, 16]);
        addComputePass(device, commandEncoder, pipelines[465], layouts[465], infinityBuf, [buf_381, buf_377], [128, 2, 32]);
        addComputePass(device, commandEncoder, pipelines[466], layouts[466], infinityBuf, [buf_377, buf_381], [16, 2, 256]);
        addComputePass(device, commandEncoder, pipelines[467], layouts[467], infinityBuf, [buf_381, buf_377], [32, 2, 64]);
        addComputePass(device, commandEncoder, pipelines[468], layouts[468], infinityBuf, [buf_377, buf_381], [2, 1024, 1]);
        addComputePass(device, commandEncoder, pipelines[469], layouts[469], infinityBuf, [buf_381, buf_377], [2, 2048, 1]);
        addComputePass(device, commandEncoder, pipelines[470], layouts[470], infinityBuf, [buf_377, buf_381], [2, 4096, 1]);
        addComputePass(device, commandEncoder, pipelines[471], layouts[471], infinityBuf, [buf_381, buf_377], [2, 2, 2048]);
        addComputePass(device, commandEncoder, pipelines[472], layouts[472], infinityBuf, [buf_377, buf_381], [2, 8192, 1]);
        addComputePass(device, commandEncoder, pipelines[473], layouts[473], infinityBuf, [buf_381, buf_377], [2048, 2, 4]);
        addComputePass(device, commandEncoder, pipelines[474], layouts[474], infinityBuf, [buf_377, buf_381], [2048, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[475], layouts[475], infinityBuf, [buf_381, buf_377], [1024, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[476], layouts[476], infinityBuf, [buf_377, buf_381], [512, 2, 8]);
        addComputePass(device, commandEncoder, pipelines[477], layouts[477], infinityBuf, [buf_381, buf_377], [256, 2, 16]);
        addComputePass(device, commandEncoder, pipelines[478], layouts[478], infinityBuf, [buf_377, buf_381], [128, 2, 32]);
        addComputePass(device, commandEncoder, pipelines[479], layouts[479], infinityBuf, [buf_381, buf_377], [16, 2, 256]);
        addComputePass(device, commandEncoder, pipelines[480], layouts[480], infinityBuf, [buf_377, buf_381], [32, 2, 64]);
        addComputePass(device, commandEncoder, pipelines[481], layouts[481], infinityBuf, [buf_381, buf_377], [2, 1024, 1]);
        addComputePass(device, commandEncoder, pipelines[482], layouts[482], infinityBuf, [buf_377, buf_381], [2, 2048, 1]);
        addComputePass(device, commandEncoder, pipelines[483], layouts[483], infinityBuf, [buf_381, buf_377], [2, 4096, 1]);
        addComputePass(device, commandEncoder, pipelines[484], layouts[484], infinityBuf, [buf_377, buf_381], [2, 2, 2048]);
        addComputePass(device, commandEncoder, pipelines[485], layouts[485], infinityBuf, [buf_381, buf_377], [2, 8192, 1]);
        addComputePass(device, commandEncoder, pipelines[486], layouts[486], infinityBuf, [buf_377, buf_381], [2048, 2, 2]);
        addComputePass(device, commandEncoder, pipelines[487], layouts[487], infinityBuf, [buf_381, buf_377], [4096, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[488], layouts[488], infinityBuf, [buf_377, buf_381], [2048, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[489], layouts[489], infinityBuf, [buf_381, buf_377], [1024, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[490], layouts[490], infinityBuf, [buf_377, buf_381], [512, 2, 8]);
        addComputePass(device, commandEncoder, pipelines[491], layouts[491], infinityBuf, [buf_381, buf_377], [256, 2, 16]);
        addComputePass(device, commandEncoder, pipelines[492], layouts[492], infinityBuf, [buf_377, buf_381], [128, 2, 32]);
        addComputePass(device, commandEncoder, pipelines[493], layouts[493], infinityBuf, [buf_381, buf_377], [16, 2, 256]);
        addComputePass(device, commandEncoder, pipelines[494], layouts[494], infinityBuf, [buf_377, buf_381], [32, 2, 64]);
        addComputePass(device, commandEncoder, pipelines[495], layouts[495], infinityBuf, [buf_381, buf_377], [2, 1024, 1]);
        addComputePass(device, commandEncoder, pipelines[496], layouts[496], infinityBuf, [buf_377, buf_381], [2, 2048, 1]);
        addComputePass(device, commandEncoder, pipelines[497], layouts[497], infinityBuf, [buf_381, buf_377], [2, 4096, 1]);
        addComputePass(device, commandEncoder, pipelines[498], layouts[498], infinityBuf, [buf_377, buf_381], [2, 2, 2048]);
        addComputePass(device, commandEncoder, pipelines[499], layouts[499], infinityBuf, [buf_381, buf_377], [2, 8192, 1]);
        addComputePass(device, commandEncoder, pipelines[500], layouts[500], infinityBuf, [buf_377, buf_381], [2048, 2, 2]);
        addComputePass(device, commandEncoder, pipelines[501], layouts[501], infinityBuf, [buf_381, buf_377], [2048, 2, 2]);
        addComputePass(device, commandEncoder, pipelines[502], layouts[502], infinityBuf, [buf_377, buf_381], [4096, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[503], layouts[503], infinityBuf, [buf_381, buf_377], [2048, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[504], layouts[504], infinityBuf, [buf_377, buf_381], [1024, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[505], layouts[505], infinityBuf, [buf_381, buf_377], [512, 2, 8]);
        addComputePass(device, commandEncoder, pipelines[506], layouts[506], infinityBuf, [buf_377, buf_381], [256, 2, 16]);
        addComputePass(device, commandEncoder, pipelines[507], layouts[507], infinityBuf, [buf_381, buf_377], [128, 2, 32]);
        addComputePass(device, commandEncoder, pipelines[508], layouts[508], infinityBuf, [buf_377, buf_381], [16, 2, 256]);
        addComputePass(device, commandEncoder, pipelines[509], layouts[509], infinityBuf, [buf_381, buf_377], [32, 2, 64]);
        addComputePass(device, commandEncoder, pipelines[510], layouts[510], infinityBuf, [buf_377, buf_381], [2, 1024, 1]);
        addComputePass(device, commandEncoder, pipelines[511], layouts[511], infinityBuf, [buf_381, buf_377], [2, 2048, 1]);
        addComputePass(device, commandEncoder, pipelines[512], layouts[512], infinityBuf, [buf_377, buf_381], [2, 4096, 1]);
        addComputePass(device, commandEncoder, pipelines[513], layouts[513], infinityBuf, [buf_381, buf_377], [2, 2, 2048]);
        addComputePass(device, commandEncoder, pipelines[514], layouts[514], infinityBuf, [buf_377, buf_381], [2, 8192, 1]);
        addComputePass(device, commandEncoder, pipelines[515], layouts[515], infinityBuf, [buf_381, buf_377], [4096, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[516], layouts[516], infinityBuf, [buf_377, buf_381], [2048, 2, 2]);
        addComputePass(device, commandEncoder, pipelines[517], layouts[517], infinityBuf, [buf_381, buf_377], [4096, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[518], layouts[518], infinityBuf, [buf_377, buf_381], [2048, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[519], layouts[519], infinityBuf, [buf_381, buf_377], [1024, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[520], layouts[520], infinityBuf, [buf_377, buf_381], [512, 2, 8]);
        addComputePass(device, commandEncoder, pipelines[521], layouts[521], infinityBuf, [buf_381, buf_377], [256, 2, 16]);
        addComputePass(device, commandEncoder, pipelines[522], layouts[522], infinityBuf, [buf_377, buf_381], [128, 2, 32]);
        addComputePass(device, commandEncoder, pipelines[523], layouts[523], infinityBuf, [buf_381, buf_377], [16, 2, 256]);
        addComputePass(device, commandEncoder, pipelines[524], layouts[524], infinityBuf, [buf_377, buf_381], [32, 2, 64]);
        addComputePass(device, commandEncoder, pipelines[525], layouts[525], infinityBuf, [buf_381, buf_377], [2, 1024, 1]);
        addComputePass(device, commandEncoder, pipelines[526], layouts[526], infinityBuf, [buf_377, buf_381], [2, 2048, 1]);
        addComputePass(device, commandEncoder, pipelines[527], layouts[527], infinityBuf, [buf_381, buf_377], [2, 4096, 1]);
        addComputePass(device, commandEncoder, pipelines[528], layouts[528], infinityBuf, [buf_377, buf_381], [2, 2, 2048]);
        addComputePass(device, commandEncoder, pipelines[529], layouts[529], infinityBuf, [buf_381, buf_377], [2, 8192, 1]);
        addComputePass(device, commandEncoder, pipelines[530], layouts[530], infinityBuf, [buf_384, buf_381], [1950, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[531], layouts[531], infinityBuf, [buf_385, buf_372, buf_381, buf_378, buf_384], [75, 1, 1]);
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
