// Optional browser smoke check. Uses Node 22+ and an installed Chrome; no npm packages.
// Start CODE/web_dashboard.py --no-browser first, then run this file with node.
import {spawn} from 'node:child_process';
import {mkdtemp, readFile, readdir, mkdir, writeFile, rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import path from 'node:path';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';

const profile = await mkdtemp(path.join(tmpdir(), 'mediroute-browser-'));
const chromePath = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const output = path.resolve('results/verification/web_dashboard');
const baseUrl = process.env.DASHBOARD_URL || 'http://127.0.0.1:8000';
const testedVirtualCount = process.env.DASHBOARD_VIRTUAL_COUNT == null ? null : Number(process.env.DASHBOARD_VIRTUAL_COUNT);
const testedTraffic = Number(process.env.DASHBOARD_TRAFFIC || 0);
await mkdir(output, {recursive: true});
const chrome = spawn(chromePath, ['--headless=new', '--no-sandbox', '--disable-gpu', '--no-first-run', '--no-default-browser-check',
  '--remote-debugging-port=0', `--user-data-dir=${profile}`, 'about:blank'], {windowsHide: true, stdio: 'ignore'});
const errors = [], pending = new Map();
const captureOnly = process.argv.includes('--capture-only');
let socket, nextId = 0, ownedRun = false;
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
async function until(callback, label, timeout = 120000) {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    const result = await callback();
    if (result) return result;
    await delay(250);
  }
  throw new Error(`Timed out: ${label}`);
}
function call(method, params = {}) {
  const id = ++nextId;
  return new Promise((resolve, reject) => {
    pending.set(id, {resolve, reject});
    socket.send(JSON.stringify({id, method, params}));
  });
}
async function evaluate(expression) {
  const result = await call('Runtime.evaluate', {expression, awaitPromise: true, returnByValue: true});
  if (result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
  return result.result.value;
}
async function screenshot(name) {
  const {data} = await call('Page.captureScreenshot', {format: 'png', captureBeyondViewport: true});
  await writeFile(path.join(output, name), Buffer.from(data, 'base64'));
}
async function checkHospitalCongestion() {
  const config = await evaluate('api("/api/config")');
  assert.equal(config.hospital_congestion.mode, 'balanced');
  const model = config.models.find(m => m.group === 'v6_s6_updates_s42') || config.models.find(m => m.virtual_count === 2);
  assert.ok(model);
  const cases = [];
  for (const [name, level] of [['low', 0], ['high', 2]]) {
    await until(() => evaluate('!document.getElementById("start").disabled'), 'ready');
    await evaluate(`(() => {
      document.getElementById('reset-settings').click();
      document.getElementById('model').value=${JSON.stringify(model.path)};
      document.getElementById('model').dispatchEvent(new Event('change'));
      for(const [id,value] of Object.entries({virtual_count:2,duration:30,patient_count:1})) {
        document.getElementById(id).value=value;document.getElementById(id).dispatchEvent(new Event('input'));
      }
      document.querySelector('[data-setting="traffic"] [data-level="0"]').click();
      document.querySelector('[data-setting="occupancy"] [data-level="${level}"]').click();
      document.getElementById('demand_mode').value='count';document.getElementById('demand_mode').dispatchEvent(new Event('change'));
      document.getElementById('speed').value='300';document.getElementById('speed').dispatchEvent(new Event('change'));
    })()`);
    await delay(350);
    const previousRun = await evaluate('currentState.run_id');
    await evaluate('document.getElementById("start").click()');
    await until(() => evaluate(`currentState.run_id!==${JSON.stringify(previousRun)} && ["completed","error"].includes(currentState.status)`), 'congestion run', 180000);
    assert.equal(await evaluate('currentState.status'), 'completed', await evaluate('currentState.message'));
    const state = await evaluate('currentState');
    assert.equal(state.result.congestion_reward_mode, 'balanced');
    const report = await evaluate(`api('/api/patients?run=${state.run_id}&limit=100')`);
    assert.equal(report.rows.length, 1);
    const patient = report.rows[0], ratio = patient.occupancy_at_selection;
    const expected = 8 * (1 - 2 * ratio) - 5 * Math.max(0, ratio - .7) / .3;
    assert.equal(patient.arrived, true);
    assert.equal(patient.congestion_reward_mode, 'balanced');
    assert.ok(Math.abs(patient.hospital_congestion_reward - expected) < 1e-8);
    assert.ok(name === 'low' ? expected > 0 : expected < 0);
    const persisted = JSON.parse(await readFile(path.join('results/web_dashboard', state.run_id, 'run.json'), 'utf8'));
    assert.equal(persisted.hospital_congestion.mode, 'balanced');
    await delay(800);
    assert.ok(await evaluate('document.querySelectorAll(".hospital-congestion-score").length>0'));
    assert.ok(await evaluate('document.getElementById("reward-components").textContent.includes("병원별 혼잡도")'));
    await screenshot(`hospital-congestion-${name}-desktop.png`);
    await evaluate('document.getElementById("result-run").value="";document.getElementById("result-run").dispatchEvent(new Event("change"))');
    await until(() => evaluate('document.querySelectorAll(".patient-detail-button").length===1'), 'patient row');
    await evaluate('document.querySelector(".patient-detail-button").click()');
    assert.ok(await evaluate('document.querySelector(".congestion-detail").textContent.includes("실제 적용 병원 혼잡도 보상")'));
    assert.equal(await evaluate('Boolean(document.querySelector(".congestion-detail strong.positive"))'), name === 'low');
    await screenshot(`hospital-congestion-${name}-patient.png`);
    await evaluate('document.getElementById("close-patient-dialog").click()');
    await call('Emulation.setDeviceMetricsOverride', {width:390,height:844,deviceScaleFactor:1,mobile:true});
    await delay(250);
    assert.ok(await evaluate('document.documentElement.scrollWidth<=innerWidth'));
    assert.ok(await evaluate('document.getElementById("hospital-list").scrollHeight<=document.getElementById("hospital-list").clientHeight+1'));
    await screenshot(`hospital-congestion-${name}-mobile.png`);
    await call('Emulation.setDeviceMetricsOverride', {width:1440,height:1200,deviceScaleFactor:1,mobile:false});
    const csv = await evaluate('(async()=>{const r=await fetch(document.getElementById("patient-csv").href);return r.text()})()');
    assert.ok(csv.includes('실제 적용 병원 혼잡도 보상'));
    cases.push({name,runId:state.run_id,hospital:patient.hospital_name,occupancy:ratio,actualCongestionReward:expected});
  }
  assert.deepEqual(errors, []);
  await writeFile(path.join(output, 'hospital-congestion-check.json'), JSON.stringify({passed:true,cases,
    detailsAndCsv:true,profilePersisted:true,mobileWidth:390,runtimeErrors:errors}, null, 2));
}
async function checkHospitalRejections() {
  await until(() => evaluate('!document.getElementById("start").disabled'), 'dashboard ready');
  const config=await evaluate('api("/api/config")');
  assert.equal(config.hospital_admission.rejection_penalty,-12);
  assert.equal(config.hospital_admission.stage,'after_arrival');
  assert.equal(config.hospital_admission.pre_departure_check,true);
  assert.equal(config.hospital_admission.reserve_during_transport,false);
  const model=config.models.find(m=>m.group==='v6_s6_updates_s42') || config.models.find(m=>m.hospital_resize_supported);
  await evaluate(`(() => {
    document.getElementById('reset-settings').click();
    document.getElementById('model').value=${JSON.stringify(model.path)};
    document.getElementById('model').dispatchEvent(new Event('change'));
    for(const [id,value] of Object.entries({virtual_count:2,duration:30,patient_count:30})) {
      document.getElementById(id).value=value;document.getElementById(id).dispatchEvent(new Event('input'));
    }
    document.querySelector('[data-setting="traffic"] [data-level="0"]').click();
    document.querySelector('[data-setting="occupancy"] [data-level="2"]').click();
    document.getElementById('demand_mode').value='count';document.getElementById('demand_mode').dispatchEvent(new Event('change'));
    document.getElementById('speed').value='300';document.getElementById('speed').dispatchEvent(new Event('change'));
  })()`);
  assert.equal(await evaluate('document.getElementById("hospital_rejection").checked'),true);
  await delay(350);
  await evaluate('document.getElementById("start").click()');
  await until(() => evaluate('currentState.status === "error" || (currentState.frame?.rejection_count || 0)>0'), 'actual hospital refusal',180000);
  assert.notEqual(await evaluate('currentState.status'),'error',await evaluate('currentState.message'));
  assert.equal(await evaluate('currentState.hospital_admission_mode'),'arrival');
  assert.equal(await evaluate('document.getElementById("hospital_rejection").disabled'),true);
  await evaluate('document.getElementById("pause").click()');
  await until(() => evaluate('currentState.status === "paused"'), 'pause after actual rejection');
  await delay(500);
  const live=await evaluate('currentState');
  assert.ok(live.frame.rejection_count>0);
  assert.equal(live.frame.reward_components.rejection,-12*live.frame.rejection_count);
  assert.ok(Math.abs(Object.values(live.frame.reward_components).reduce((a,b)=>a+b,0)-live.frame.reward)<1e-7);
  assert.ok(await evaluate('document.getElementById("reward-components").textContent.includes("수용 거절")'));
  await screenshot('desktop-arrival-rejections.png');
  await call('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
  await delay(350);
  assert.ok(await evaluate('document.documentElement.scrollWidth<=390'));
  await screenshot('mobile-arrival-rejections.png');
  await call('Emulation.setDeviceMetricsOverride',{width:1440,height:1200,deviceScaleFactor:1,mobile:false});
  await evaluate('document.getElementById("pause").click()');
  const patient=await until(async () => {
    assert.notEqual(await evaluate('currentState.status'),'error',await evaluate('currentState.message'));
    const report=await evaluate('api("/api/patients?limit=100")');
    return report.rows.find(row=>row.rejection_count>0 && row.hospital_visits?.length>1 && row.arrived);
  },'patient with an actual refusal history',180000);
  assert.equal(patient.reward_components.rejection,-12*patient.rejection_count);
  assert.equal(patient.rejection_history.length,patient.rejection_count);
  assert.ok(patient.rejection_history.every(event=>event.stage==='after_arrival' && event.time>event.leg_dispatch_time && event.leg_dispatch_time>=event.dispatch_time));
  assert.ok(patient.hospital_visits.every(visit=>visit.vehicle_id===patient.vehicle_id));
  await evaluate(`document.getElementById('filter-search').value=${JSON.stringify(patient.patient_id)};document.getElementById('filter-search').dispatchEvent(new Event('input'))`);
  await until(() => evaluate(`Array.from(document.querySelectorAll('.patient-detail-button')).some(button=>button.dataset.record===${JSON.stringify(patient.record_id)})`),'rejected patient in table');
  await evaluate(`Array.from(document.querySelectorAll('.patient-detail-button')).find(button=>button.dataset.record===${JSON.stringify(patient.record_id)}).click()`);
  assert.equal(await evaluate('document.querySelectorAll(".rejection-history li").length'),patient.rejection_count);
  assert.ok(await evaluate('document.getElementById("patient-dialog-content").textContent.includes("병원 수용 거절 이력")'));
  assert.equal(await evaluate('document.querySelectorAll(".hospital-visit-history li").length'),patient.hospital_visits.length);
  assert.ok(await evaluate('document.getElementById("patient-dialog-content").textContent.includes("도착 후 거절")'));
  await screenshot('patient-arrival-rejection-history.png');
  const csv=await evaluate('(async()=>{const r=await fetch(document.getElementById("patient-csv").href);return r.text()})()');
  assert.ok(csv.includes('병원 수용 거절 횟수') && csv.includes('병원 거절 이력(JSON)'));
  assert.ok(csv.includes('실제 병원 방문 이력(JSON)'));
  const exported=await evaluate('api("/api/export")');
  assert.ok(exported.patient_outcomes.some(row=>row.record_id===patient.record_id && row.rejection_count===patient.rejection_count));
  await evaluate('document.getElementById("close-patient-dialog").click()');
  if(await evaluate('["running","paused"].includes(currentState.status)')) {
    await evaluate('document.getElementById("stop").click()');
    await until(() => evaluate('currentState.status === "stopped"'),'stop rejection test');
  }
  await until(() => evaluate('!document.getElementById("start").disabled'),'worker cleanup');
  const requestRun=await evaluate('currentState.run_id');
  await evaluate(`document.getElementById('hospital_rejection').checked=false;document.getElementById('patient_count').value=0;document.getElementById('patient_count').dispatchEvent(new Event('input'));document.getElementById('start').click()`);
  await until(() => evaluate(`currentState.run_id!==${JSON.stringify(requestRun)} && ['completed','error'].includes(currentState.status)`),'legacy admission comparison',120000);
  assert.equal(await evaluate('currentState.status'),'completed',await evaluate('currentState.message'));
  assert.equal(await evaluate('currentState.result.hospital_admission_mode'),'precheck');
  assert.equal(await evaluate('currentState.result.rejection_count'),0);
  assert.deepEqual(errors,[]);
  await writeFile(path.join(output,'arrival-rejection-check.json'),JSON.stringify({passed:true,
    runId:requestRun,liveRefusals:live.frame.rejection_count,patient:patient.patient_id,
    patientRefusals:patient.rejection_count,patientPenalty:patient.reward_components.rejection,
    history:patient.rejection_history,visits:patient.hospital_visits,sameVehicle:true,
    onlyAfterArrival:true,preDepartureCheck:true,detailsAndCsv:true,liveRewardAccounting:true,
    legacyToggle:true,mobileWidth:390,runtimeErrors:errors},null,2));
}
async function checkHospitalMix() {
  await until(() => evaluate('!document.getElementById("start").disabled'), 'dashboard ready');
  const config=await evaluate('api("/api/config")');
  const model=config.models.find(m=>m.group==='v6_s6_updates_s42' && m.hospital_resize_supported)
    || config.models.find(m=>m.hospital_resize_supported);
  assert.ok(model);
  const hashBefore=createHash('sha256').update(await readFile(path.resolve(model.path))).digest('hex');
  await evaluate(`(() => {
    document.getElementById('reset-settings').click();
    document.getElementById('model').value=${JSON.stringify(model.path)};
    document.getElementById('model').dispatchEvent(new Event('change'));
    document.getElementById('virtual_count').value=10;
    document.getElementById('virtual_count').dispatchEvent(new Event('input'));
  })()`);
  assert.deepEqual(await evaluate('virtualMixPlan(10)'),[2,4,3,1]);
  await evaluate(`document.getElementById('virtual_count').value=1;document.getElementById('virtual_count').dispatchEvent(new Event('input'))`);
  assert.equal(await evaluate('document.getElementById("virtual_count").checkValidity()'),false);
  await evaluate(`(() => {
    document.getElementById('virtual_count').value=5;document.getElementById('virtual_count').dispatchEvent(new Event('input'));
    for(const key of virtualTypeKeys){document.getElementById('virtual_ratio_'+key).value=0;document.getElementById('virtual_ratio_'+key).dispatchEvent(new Event('input'));}
  })()`);
  assert.equal(await evaluate('document.getElementById("virtual_ratio_local_center").checkValidity()'),false);
  await evaluate(`(() => {
    for(const key of virtualTypeKeys){document.getElementById('virtual_ratio_'+key).value=1;document.getElementById('virtual_ratio_'+key).dispatchEvent(new Event('input'));}
    for(const [id,value] of Object.entries({duration:30,patient_count:1})) {document.getElementById(id).value=value;document.getElementById(id).dispatchEvent(new Event('input'));}
    document.querySelector('[data-setting="traffic"] [data-level="0"]').click();
    document.getElementById('demand_mode').value='count';document.getElementById('demand_mode').dispatchEvent(new Event('change'));
  })()`);
  assert.deepEqual(await evaluate('virtualMixPlan(5)'),[2,1,1,1]);
  await screenshot('desktop-virtual-hospital-mix.png');
  await call('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
  await delay(350);
  assert.equal(await evaluate('document.documentElement.scrollWidth <= 390'),true);
  await screenshot('mobile-virtual-hospital-mix.png');
  await call('Emulation.setDeviceMetricsOverride',{width:1440,height:1200,deviceScaleFactor:1,mobile:false});
  await evaluate(`document.getElementById('speed').value='300';document.getElementById('speed').dispatchEvent(new Event('change'))`);
  await delay(350);
  await evaluate('document.getElementById("start").click()');
  await until(() => evaluate('currentState.status === "error" || Boolean(currentState.virtual_hospital_mix)'), 'native mixed hospitals ready',180000);
  assert.notEqual(await evaluate('currentState.status'),'error',await evaluate('currentState.message'));
  const expected={tertiary:2,local_center:1,local_institution:1,reported_er:1};
  assert.deepEqual(await evaluate('currentState.virtual_hospital_mix.counts'),expected);
  if(await evaluate('["preparing","running","paused"].includes(currentState.status)')) {
    assert.equal(await evaluate('document.getElementById("virtual_ratio_local_center").disabled'),true);
  }
  await until(() => evaluate('["completed","error"].includes(currentState.status)'), 'mixed hospital simulation complete',180000);
  assert.equal(await evaluate('currentState.status'),'completed',await evaluate('currentState.message'));
  const state=await evaluate('currentState');
  const labels=['권역응급의료센터','지역응급의료센터','지역응급의료기관','응급실운영신고기관'];
  const actual=labels.map(type=>state.hospitals.filter(h=>h.virtual && h.type===type).length);
  assert.deepEqual(actual,[2,1,1,1]);
  assert.equal(state.result.generated_patients,1);
  assert.deepEqual(state.result.virtual_hospital_mix.counts,expected);
  const saved=JSON.parse(await readFile(path.resolve(state.output,'run.json'),'utf8'));
  assert.deepEqual(saved.options.virtual_mix,{local_center:1,local_institution:1,reported_er:1});
  assert.deepEqual(saved.virtual_hospital_mix.counts,expected);
  assert.ok(await evaluate('document.getElementById("virtual-mix-actual").textContent.includes("권역 2개")'));
  await call('Page.reload');
  await until(() => evaluate('typeof currentState !== "undefined" && currentState.options?.virtual_mix?.local_center === 1 && Boolean(token)'), 'mixture restored');
  assert.equal(await evaluate('document.getElementById("virtual_count").value'),'5');
  assert.deepEqual(await evaluate('virtualMixWeights()'),{local_center:1,local_institution:1,reported_er:1});
  await evaluate(`document.getElementById('virtual_count').value=0;document.getElementById('virtual_count').dispatchEvent(new Event('input'));document.getElementById('virtual-plus').click()`);
  assert.equal(await evaluate('document.getElementById("virtual_count").value'),'2');
  assert.deepEqual(await evaluate('virtualMixPlan(2)'),[2,0,0,0]);
  const hashAfter=createHash('sha256').update(await readFile(path.resolve(model.path))).digest('hex');
  assert.equal(hashAfter,hashBefore);
  assert.deepEqual(errors,[]);
  await writeFile(path.join(output,'hospital-mix-check.json'),JSON.stringify({passed:true,runId:state.run_id,
    totalVirtual:5,weights:saved.options.virtual_mix,expectedCounts:expected,actualCounts:actual,
    patientCount:state.result.generated_patients,defaultTenPreview:[2,4,3,1],minimumTwo:true,
    settingsRestored:true,mobileWidth:390,sourceWeightsUnchanged:true,runtimeErrors:errors},null,2));
}
async function checkTrackingAndHospitals() {
  await until(() => evaluate('!document.getElementById("start").disabled'), 'dashboard ready');
  const config = await evaluate('api("/api/config")');
  const model = config.models.find(m => m.group === 'v6_s6_updates_s42' && m.hospital_resize_supported)
    || config.models.find(m => m.hospital_resize_supported);
  assert.ok(model, 'a shared DQN model is required');
  const originalHash = createHash('sha256').update(await readFile(path.resolve(model.path))).digest('hex');
  await evaluate(`(() => {
    document.getElementById('reset-settings').click();
    document.getElementById('model').value=${JSON.stringify(model.path)};
    document.getElementById('model').dispatchEvent(new Event('change'));
    for (const [id,value] of Object.entries({virtual_count:0,duration:120,patient_count:5})) {
      document.getElementById(id).value=value; document.getElementById(id).dispatchEvent(new Event('input'));
    }
    document.querySelector('[data-setting="traffic"] [data-level="0"]').click();
    document.getElementById('demand_mode').value='count'; document.getElementById('demand_mode').dispatchEvent(new Event('change'));
    document.getElementById('speed').value='60'; document.getElementById('speed').dispatchEvent(new Event('change'));
  })()`);
  assert.equal(await evaluate('document.getElementById("virtual_count").disabled'), false);
  await delay(350);
  await evaluate('document.getElementById("start").click()');
  await until(() => evaluate('currentState.status === "error" || (currentState.frame?.vehicles?.length || 0) >= 2'), 'multiple real ambulances with resized DQN', 180000);
  assert.notEqual(await evaluate('currentState.status'), 'error', await evaluate('currentState.message'));
  assert.equal(await evaluate('currentState.virtual_count'), 0);
  assert.equal(await evaluate('currentState.hospitals.filter(h=>h.virtual).length'), 0);
  assert.equal(await evaluate('currentState.hospital_transfer.method'), 'shared_scorer_weights_unchanged');
  await evaluate('document.getElementById("pause").click()');
  await until(() => evaluate('currentState.status === "paused"'), 'pause for camera selection');
  await delay(400);
  const ids = await evaluate('vehicles.slice(0,2).map(v=>v.id)');
  assert.equal(ids.length, 2);
  await evaluate('document.getElementById("focus-ambulance").click()');
  assert.equal(await evaluate('document.getElementById("ambulance-focus-panel").hidden'), false);
  assert.ok(await evaluate('Array.from(document.getElementById("follow-ambulance").options).slice(1).every(o=>o.textContent.startsWith("구급차 "))'));
  async function select(id) {
    await evaluate(`document.getElementById('follow-ambulance').value=${JSON.stringify(id)}; document.getElementById('follow-ambulance').dispatchEvent(new Event('change'))`);
    await delay(100);
    assert.equal(await evaluate('followedVehicleId'), id);
  }
  async function camera() {
    return evaluate(`new Promise(resolve => requestAnimationFrame(now => {
      const vehicle=vehicles.find(v=>v.id===followedVehicleId);
      if(!vehicle){resolve(null);return;}
      const position=interpolatedPosition(vehicle,now);
      resolve({x:view.x,y:view.y,error:Math.hypot(view.x-position.x,view.y-position.y)});
    }))`);
  }
  await select(ids[0]);
  assert.ok((await camera()).error < .001, 'camera centers first selected ambulance');
  await select(ids[1]);
  assert.ok((await camera()).error < .001, 'camera switches to second ambulance');
  await evaluate(`document.getElementById('speed').value='10'; document.getElementById('speed').dispatchEvent(new Event('change'))`);
  const before = await camera();
  await evaluate('document.getElementById("pause").click()');
  await until(async () => {const next=await camera();return next && Math.hypot(next.x-before.x,next.y-before.y)>1;}, 'camera follows actual movement', 30000);
  const moving = await camera();
  assert.ok(moving.error < .001);
  await evaluate('document.getElementById("pause").click()');
  await until(() => evaluate('currentState.status === "paused"'), 'pause tracking');
  await delay(350);
  await screenshot('desktop-ambulance-tracking.png');
  await call('Emulation.setDeviceMetricsOverride', {width:390,height:844,deviceScaleFactor:1,mobile:true});
  await delay(350);
  assert.equal(await evaluate('document.documentElement.scrollWidth <= innerWidth'), true);
  assert.ok((await camera()).error < .001);
  await screenshot('mobile-ambulance-tracking.png');
  await call('Emulation.setDeviceMetricsOverride', {width:1440,height:1200,deviceScaleFactor:1,mobile:false});
  await evaluate('document.getElementById("follow-cancel").click()');
  assert.equal(await evaluate('followedVehicleId'), null);
  const free = await evaluate('({x:view.x,y:view.y})');
  await evaluate('document.getElementById("pause").click()');
  await delay(600);
  assert.deepEqual(await evaluate('({x:view.x,y:view.y})'), free, 'free camera does not follow');
  const liveId = await evaluate('vehicles.at(-1)?.id');
  assert.ok(liveId);
  await select(liveId);
  await evaluate('document.getElementById("zoom-in").click()');
  assert.equal(await evaluate('followedVehicleId'), liveId, 'zoom preserves tracking');
  await evaluate('document.getElementById("map-reset").click()');
  assert.equal(await evaluate('followedVehicleId'), null);
  await select(liveId);
  const point = await evaluate('(() => {const r=document.getElementById("map").getBoundingClientRect();return {x:r.x+r.width*.7,y:r.y+r.height*.7};})()');
  await call('Input.dispatchMouseEvent',{type:'mousePressed',x:point.x,y:point.y,button:'left',clickCount:1});
  await call('Input.dispatchMouseEvent',{type:'mouseMoved',x:point.x+40,y:point.y+20,buttons:1});
  await call('Input.dispatchMouseEvent',{type:'mouseReleased',x:point.x+40,y:point.y+20,button:'left'});
  assert.equal(await evaluate('followedVehicleId'), null, 'drag cancels tracking');
  await select(await evaluate('vehicles.at(-1).id'));
  await evaluate('document.getElementById("stop").click()');
  await until(() => evaluate('currentState.status === "stopped"'), 'stop tracked run');
  assert.equal(await evaluate('followedVehicleId'), null);
  await until(() => evaluate('!document.getElementById("start").disabled'), 'worker cleanup');
  await evaluate(`(() => {
    for(const [id,value] of Object.entries({virtual_count:3,duration:30,patient_count:1})) {
      document.getElementById(id).value=value;document.getElementById(id).dispatchEvent(new Event('input'));
    }
    document.getElementById('speed').value='30';document.getElementById('speed').dispatchEvent(new Event('change'));
  })()`);
  await delay(350);
  await evaluate('document.getElementById("start").click()');
  await until(() => evaluate('currentState.status === "error" || (currentState.frame?.vehicles?.length || 0) > 0'), 'DQN with added hospitals', 180000);
  assert.notEqual(await evaluate('currentState.status'), 'error', await evaluate('currentState.message'));
  assert.equal(await evaluate('currentState.virtual_count'), 3);
  assert.equal(await evaluate('currentState.hospitals.filter(h=>h.virtual).length'), 3);
  await select(await evaluate('vehicles[0].id'));
  await evaluate(`document.getElementById('speed').value='300';document.getElementById('speed').dispatchEvent(new Event('change'))`);
  await until(() => evaluate('["completed","error"].includes(currentState.status)'), 'natural mission completion', 180000);
  assert.equal(await evaluate('currentState.status'), 'completed', await evaluate('currentState.message'));
  assert.equal(await evaluate('followedVehicleId'), null, 'arrival ends tracking');
  assert.equal(await evaluate('currentState.result.generated_patients'), 1);
  const addedRun = await evaluate('currentState');
  const saved = JSON.parse(await readFile(path.resolve(addedRun.output,'run.json'),'utf8'));
  assert.equal(saved.hospital_transfer.requested_virtual_count,3);
  assert.equal(saved.hospitals.filter(h=>h.virtual).length,3);
  const finalHash=createHash('sha256').update(await readFile(path.resolve(model.path))).digest('hex');
  assert.equal(finalHash,originalHash,'source model file is unchanged');
  assert.deepEqual(errors, []);
  await writeFile(path.join(output,'hospital-tracking-check.json'),JSON.stringify({passed:true,
    model:model.path,requestedVirtualCounts:[0,3],multipleAmbulances:true,selectedIds:ids,
    cameraFollowsMovement:true,cameraSwitches:true,cancelAndDrag:true,zoomPreservesTracking:true,
    naturalArrivalStopsTracking:true,desktopWidth:1440,mobileWidth:390,sourceWeightsUnchanged:true,
    addedRun:addedRun.run_id,runtimeErrors:errors},null,2));
}
async function checkCustomDemand() {
  assert.ok(!['preparing','running','paused','stopping'].includes(await evaluate('currentState.status')));
  await evaluate('document.getElementById("reset-settings").click()');
  assert.equal(await evaluate('document.getElementById("demand_mode").value'), 'preset');
  await evaluate(`(() => {
    document.getElementById('demand_mode').value='rate';
    document.getElementById('demand_mode').dispatchEvent(new Event('change'));
    document.getElementById('custom_patients_per_hour').value=12.5;
    document.getElementById('custom_patients_per_hour').dispatchEvent(new Event('input'));
  })()`);
  assert.ok(await evaluate('document.getElementById("demand-estimate").textContent.includes("12.50")'));
  assert.equal(await evaluate('document.getElementById("custom_patients_per_hour").disabled'), false);
  await evaluate(`document.getElementById('custom_patients_per_hour').value=3601`);
  assert.equal(await evaluate('document.getElementById("custom_patients_per_hour").checkValidity()'), false);
  await evaluate(`(() => {
    document.getElementById('duration').value=60;
    document.getElementById('duration').dispatchEvent(new Event('input'));
    document.getElementById('demand_mode').value='count';
    document.getElementById('demand_mode').dispatchEvent(new Event('change'));
    document.getElementById('patient_count').value=61;
  })()`);
  assert.equal(await evaluate('document.getElementById("patient_count").checkValidity()'), false);
  await evaluate(`document.getElementById('patient_count').value=3; document.getElementById('patient_count').dispatchEvent(new Event('input')); document.querySelector('[data-mode="training"]').click()`);
  assert.ok(await evaluate('document.getElementById("learning-estimate").textContent.includes("총 9명")'));
  await evaluate(`document.querySelector('[data-mode="simulation"]').click()`);
  await screenshot('desktop-custom-patient-count.png');
  await call('Emulation.setDeviceMetricsOverride', {width:390,height:844,deviceScaleFactor:1,mobile:true});
  await delay(350);
  assert.equal(await evaluate('document.documentElement.scrollWidth <= innerWidth'), true);
  await screenshot('mobile-custom-patient-count.png');
  await call('Emulation.setDeviceMetricsOverride', {width:1440,height:1200,deviceScaleFactor:1,mobile:false});
  await evaluate(`document.getElementById('speed').value='300'; document.getElementById('speed').dispatchEvent(new Event('change'))`);
  await delay(350);
  await evaluate('document.getElementById("start").click()');
  await until(() => evaluate('["running","completed","error"].includes(currentState.status)'), 'custom count started');
  assert.notEqual(await evaluate('currentState.status'), 'error', await evaluate('currentState.message'));
  assert.equal(await evaluate('currentState.requested_patient_count'), 3);
  if (await evaluate('currentState.status === "running"')) {
    assert.equal(await evaluate('document.getElementById("patient_count").disabled'), true);
    assert.equal(await evaluate('document.getElementById("demand_mode").disabled'), true);
  }
  await call('Page.reload');
  await until(() => evaluate('typeof currentState !== "undefined" && currentState.options?.demand_mode === "count" && Boolean(token) && Boolean(network)'), 'custom settings restored on reload');
  assert.equal(await evaluate('document.getElementById("patient_count").value'), '3');
  assert.equal(await evaluate('document.getElementById("demand_mode").value'), 'count');
  await until(() => evaluate('["completed","error"].includes(currentState.status)'), 'exact custom count completed', 180000);
  assert.equal(await evaluate('currentState.status'), 'completed', await evaluate('currentState.message'));
  const countRun = await evaluate('currentState');
  assert.equal(countRun.result.generated_patients, 3);
  assert.equal(countRun.result.arrived + countRun.result.failed + countRun.result.censored, 3);
  const patients = await evaluate('api("/api/patients")');
  assert.equal(patients.record_count, 3);
  const saved = JSON.parse(await readFile(path.resolve(countRun.output, 'run.json'), 'utf8'));
  assert.equal(saved.options.patient_count, 3);
  assert.equal(saved.scenario.fixed_patient_count, 3);
  assert.equal(saved.scenario.demand_profile.profile_id, 'user_fixed_count_v1');
  await until(() => evaluate('!document.getElementById("start").disabled'), 'native worker cleanup');
  await evaluate(`(() => {
    document.getElementById('duration').value=30;
    document.getElementById('duration').dispatchEvent(new Event('input'));
    document.getElementById('demand_mode').value='rate';
    document.getElementById('demand_mode').dispatchEvent(new Event('change'));
    document.getElementById('custom_patients_per_hour').value=0;
    document.getElementById('custom_patients_per_hour').dispatchEvent(new Event('input'));
    document.getElementById('start').click();
  })()`);
  await until(() => evaluate(`currentState.run_id !== ${JSON.stringify(countRun.run_id)} && ['completed','error'].includes(currentState.status)`), 'zero hourly demand completed', 120000);
  assert.equal(await evaluate('currentState.status'), 'completed', await evaluate('currentState.message'));
  assert.equal(await evaluate('currentState.result.generated_patients'), 0);
  assert.equal(await evaluate('currentState.scenario.prob'), 0);
  await evaluate(`document.querySelector('[data-setting="demand"] [data-level="1"]').click()`);
  assert.equal(await evaluate('document.getElementById("demand_mode").value'), 'preset');
  assert.equal(await evaluate('document.getElementById("custom_patients_per_hour").disabled'), true);
  assert.ok(await evaluate('document.getElementById("demand-estimate").textContent.includes("2.14")'));
  assert.deepEqual(errors, []);
  await writeFile(path.join(output, 'custom-demand-check.json'), JSON.stringify({passed:true,
    fixedCount:3,actualGenerated:countRun.result.generated_patients,patientRecords:patients.record_count,
    countRun:countRun.run_id,zeroHourlyDemand:true,settingsRestored:true,mobileWidth:390,
    hourlyPreview:12.5,invalidInputRejected:true,presetRestored:true,runtimeErrors:errors}, null, 2));
}
async function checkEmptyTraining() {
  await until(() => evaluate('!document.getElementById("start").disabled'), 'previous worker cleanup');
  const checks = await evaluate(`(() => {
    document.getElementById('reset-settings').click();
    document.querySelector('[data-mode="training"]').click();
    for (const [id,value] of Object.entries({duration:30,episodes:1,virtual_count:0})) {
      document.getElementById(id).value=value;
      document.getElementById(id).dispatchEvent(new Event('input'));
    }
    const checks = {disabled:document.getElementById('start').disabled,mode:settings.mode,
      inputs:['duration','ambulances','episodes','seed','virtual_count'].map(id=>[id,document.getElementById(id).value,document.getElementById(id).checkValidity()])};
    document.getElementById('start').click();
    return checks;
  })()`);
  assert.equal(checks.disabled, false, JSON.stringify(checks));
  assert.equal(checks.mode, 'training', JSON.stringify(checks));
  assert.ok(checks.inputs.every(input => input[2]), JSON.stringify(checks));
  await until(() => evaluate('["preparing","running","completed","error"].includes(currentState.status)'), 'empty training started');
  await until(() => evaluate('currentState.status === "error" || currentState.status === "completed"'), 'empty training and validation completed', 180000);
  assert.equal(await evaluate('currentState.status'), 'completed', await evaluate('currentState.message'));
  const emptyTraining = await evaluate('currentState');
  assert.equal(emptyTraining.options.mode, 'training');
  assert.equal(emptyTraining.result.generated_patients, 0);
  assert.equal(emptyTraining.result.training_updates, 0);
  assert.equal(emptyTraining.result.validation_generated_patients, 0);
  assert.equal(emptyTraining.result.validation_avg_reward, null);
  const manifest = JSON.parse(await readFile(path.resolve(emptyTraining.output, 'experiment.json'), 'utf8'));
  assert.deepEqual(manifest.validation_seeds, [50042]);
  const models = await readdir(path.resolve(emptyTraining.output, 'models'));
  assert.ok(models.some(name => name.endsWith('_last.pth')));
  assert.ok(!models.some(name => name.endsWith('_best.pth')));
  assert.ok(await evaluate('document.getElementById("result-note").textContent.includes("아직 가중치가 학습되지 않았습니다")'));
  await writeFile(path.join(output, 'training-empty-check.json'), JSON.stringify({passed:true,runId:emptyTraining.run_id,
    result:emptyTraining.result,validationSeeds:manifest.validation_seeds,models}, null, 2));
  await screenshot('desktop-empty-training.png');
}
try {
  const port = await until(async () => {
    try { return Number((await readFile(path.join(profile, 'DevToolsActivePort'), 'utf8')).split('\n')[0]); }
    catch { return false; }
  }, 'Chrome launch', 20000);
  const targets = await until(async () => {
    try {
      const targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
      return targets.some(t => t.type === 'page') ? targets : false;
    } catch { return false; }
  }, 'Chrome debugging endpoint', 20000);
  socket = new WebSocket(targets.find(t => t.type === 'page').webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {socket.onopen = resolve; socket.onerror = reject;});
  socket.onmessage = event => {
    const message = JSON.parse(event.data);
    if (message.id) {
      const target = pending.get(message.id); pending.delete(message.id);
      if (message.error) target.reject(new Error(JSON.stringify(message.error))); else target.resolve(message.result);
    } else if (message.method === 'Runtime.exceptionThrown') errors.push(message.params.exceptionDetails);
  };
  await call('Page.enable'); await call('Runtime.enable');
  await call('Emulation.setDeviceMetricsOverride', {width: 1440, height: 1200, deviceScaleFactor: 1, mobile: false});
  await call('Page.navigate', {url: baseUrl});
  await until(() => evaluate('typeof network !== "undefined" && Boolean(network) && Boolean(token)'), 'dashboard and map');
  await until(() => evaluate('document.getElementById("connection").classList.contains("connected")'), 'dashboard state connected');
  if (captureOnly) {
    if (process.argv.includes('--mobile')) {
      await call('Emulation.setDeviceMetricsOverride', {width:390,height:844,deviceScaleFactor:1,mobile:true});
      await delay(500);
      assert.ok(await evaluate('document.documentElement.scrollWidth <= 390'), 'mobile layout fits device width');
    }
    await delay(800);
    await screenshot(process.argv.includes('--mobile') ? 'mobile-layout.png' : 'desktop-result.png');
    console.log('Captured current dashboard.');
  } else if (process.argv.includes('--congestion-only')) {
    ownedRun=true;
    await checkHospitalCongestion();
    console.log('PASS: real DQN low/high congestion bonuses/penalties, hospital previews, patient details, CSV, profiles, PC/mobile.');
  } else if (process.argv.includes('--rejections-only')) {
    ownedRun=true;
    await checkHospitalRejections();
    console.log('PASS: real refusal/reselection penalties, immediate reward accounting, details/CSV, comparison toggle, PC/mobile.');
  } else if (process.argv.includes('--hospital-mix-only')) {
    ownedRun=true;
    await checkHospitalMix();
    console.log('PASS: fixed two tertiary centers, ratio allocation, actual native types, saved settings, PC/mobile.');
  } else if (process.argv.includes('--tracking-only')) {
    ownedRun = true;
    await checkTrackingAndHospitals();
    console.log('PASS: DQN hospital counts 0/3, selected ambulance following/switching, cancel/drag/zoom, arrival cleanup, PC/mobile.');
  } else if (process.argv.includes('--custom-demand-only')) {
    ownedRun = true;
    await checkCustomDemand();
    console.log('PASS: custom hourly/count controls, exact native patient count, zero rate, saved settings, desktop/mobile.');
  } else if (process.argv.includes('--empty-training-only')) {
    ownedRun = true;
    await checkEmptyTraining();
    assert.deepEqual(errors, []);
    console.log('PASS: real empty training, no updates, no unscored best, validation seed retained.');
  } else {
  assert.ok(!['preparing','running','paused','stopping'].includes(await evaluate('currentState.status')), 'start with an idle dashboard');
  await evaluate('document.getElementById("reset-settings").click()');
  assert.ok(await evaluate('document.querySelectorAll(".hospital").length >= 7'));
  await until(() => evaluate('document.querySelectorAll(".severity-card").length === 5'), 'patient summary');
  const savedRuns = await evaluate('api("/api/runs")');
  let saved = null, savedReport = null;
  for (const candidate of savedRuns.runs.filter(run => run.completed_episodes > 0)) {
    const report = await evaluate(`api('/api/patients?run='+encodeURIComponent(${JSON.stringify(candidate.id)}))`);
    if (report.record_count > 0) { saved = candidate; savedReport = report; break; }
  }
  if (saved) {
    await evaluate(`document.getElementById('result-run').value=${JSON.stringify(saved.id)}; document.getElementById('result-run').dispatchEvent(new Event('change'))`);
    await until(() => evaluate('document.querySelectorAll(".patient-detail-button").length > 0'), 'saved patient rows');
    const report = await evaluate(`api('/api/patients?run='+encodeURIComponent(${JSON.stringify(saved.id)}))`);
    assert.equal(report.severities.reduce((n,s)=>n+s.total,0), report.record_count);
    assert.ok(report.record_count >= report.rows.length, 'saved cohort and pagination counts are consistent');
    assert.ok(report.rows.every(row => row.hospital_name || !row.hospital_id));
    const severity = savedReport.severities.find(s => s.total > 0).ktas;
    await evaluate(`document.getElementById('filter-ktas').value=${JSON.stringify(String(severity))}; document.getElementById('filter-ktas').dispatchEvent(new Event('change'))`);
    await until(() => evaluate(`Array.from(document.querySelectorAll('.patient-table .severity-badge')).every(el=>el.textContent.trim()===${JSON.stringify(`KTAS ${severity}`)})`), 'severity filter');
    assert.ok(await evaluate('document.querySelectorAll(".patient-detail-button").length > 0'));
    await evaluate('document.querySelector(".patient-detail-button").click()');
    assert.equal(await evaluate('document.getElementById("patient-dialog").open'), true);
    assert.ok(await evaluate('document.getElementById("patient-dialog-content").textContent.includes("환자별 보상 내역")'));
    await screenshot('patient-detail.png');
    await evaluate('document.getElementById("close-patient-dialog").click()');
    const csvCheck = await evaluate(`(async()=> { const r=await fetch(document.getElementById('patient-csv').href); return {ok:r.ok,type:r.headers.get('Content-Type'),text:await r.text()}; })()`);
    assert.ok(csvCheck.ok && csvCheck.type.includes('text/csv'));
    assert.ok(csvCheck.text.includes('배정 병원') && csvCheck.text.includes('보상'));
    await evaluate('document.getElementById("patient-filters").reset()');
    await delay(500);
  }
  assert.equal(await evaluate('document.documentElement.scrollWidth <= innerWidth'), true);
  assert.equal(await evaluate('Number(document.getElementById("duration").value)'), 3600);
  assert.ok(await evaluate('demandProfile.patients_per_hour[1] > 2 && demandProfile.patients_per_hour[1] < 2.2'));
  assert.ok(await evaluate('document.getElementById("demand-estimate").textContent.includes("2.14")'));
  // Release an old result inside the filter debounce window; it must never render.
  await evaluate(`(() => {
    window.originalDashboardApi = api;
    window.releaseOldReport = null;
    api = async (path, body) => {
      const report = await window.originalDashboardApi(path, body);
      if (path.startsWith('/api/patients?') && path.includes('ktas=1')) {
        report.record_count = 999999;
        await new Promise(resolve => { window.releaseOldReport = resolve; });
      }
      return report;
    };
    document.getElementById('filter-ktas').value='1';
    document.getElementById('filter-ktas').dispatchEvent(new Event('change'));
  })()`);
  await until(() => evaluate('Boolean(window.releaseOldReport)'), 'held patient report');
  const didDiscardOld = await evaluate(`(async () => {
    document.getElementById('filter-ktas').value='2';
    document.getElementById('filter-ktas').dispatchEvent(new Event('change'));
    window.releaseOldReport();
    await new Promise(resolve => setTimeout(resolve, 30));
    api = window.originalDashboardApi;
    return !document.getElementById('patient-record-count').textContent.includes('999,999');
  })()`);
  assert.equal(didDiscardOld, true, 'stale filter response is ignored immediately');
  await evaluate('document.getElementById("patient-filters").reset()');
  await delay(500);
  await evaluate(`document.querySelector('[data-mode="training"]').click()`);
  assert.equal(await evaluate('document.getElementById("start").textContent.trim()'), '학습 시작');
  assert.equal(await evaluate('document.getElementById("policy-field").hidden'), true);
  assert.equal(await evaluate('getComputedStyle(document.getElementById("policy-field")).display'), 'none');
  assert.equal(await evaluate('document.getElementById("virtual_count").disabled'), false);
  assert.equal(await evaluate('document.getElementById("learning-estimate").hidden'), false);
  assert.ok(await evaluate('document.getElementById("learning-estimate").textContent.includes("6.4")'));
  await evaluate(`document.getElementById('virtual_count').value='0'; document.getElementById('virtual_count').dispatchEvent(new Event('input'))`);
  assert.equal(await evaluate('document.getElementById("virtual-minus").disabled'), true);
  await evaluate('document.getElementById("virtual-plus").click()');
  assert.equal(await evaluate('document.getElementById("virtual_count").value'), '2');
  await evaluate(`document.getElementById('virtual_count').value='10'; document.getElementById('virtual_count').dispatchEvent(new Event('input'))`);
  assert.equal(await evaluate('document.getElementById("virtual-plus").disabled'), true);
  await evaluate(`document.getElementById('virtual_count').value='1.5'; document.getElementById('virtual_count').dispatchEvent(new Event('input'))`);
  assert.equal(await evaluate('document.getElementById("virtual_count").checkValidity()'), false);
  await evaluate(`document.getElementById('virtual_count').value='4'; document.getElementById('virtual_count').dispatchEvent(new Event('input'))`);
  await screenshot('virtual-hospital-settings.png');
  await evaluate(`document.querySelector('[data-mode="simulation"]').click()`);
  assert.equal(await evaluate('document.getElementById("virtual_count").disabled'), false, 'DQN simulation permits hospital-set changes');
  const modelCount = await evaluate('modelMetadata.get(document.getElementById("model").value)?.virtual_count');
  assert.equal(await evaluate('Number(document.getElementById("virtual_count").value)'), 4, 'manual count survives policy changes');
  await evaluate(`document.querySelector('[data-mode="training"]').click()`);
  assert.equal(await evaluate('document.getElementById("virtual_count").value'), '4', 'manual count survives a model switch');
  await evaluate(`document.querySelector('[data-mode="simulation"]').click(); document.getElementById('policy').value='shortest_time'; document.getElementById('policy').dispatchEvent(new Event('change'))`);
  assert.equal(await evaluate('document.getElementById("virtual_count").disabled'), false);
  assert.equal(await evaluate('document.getElementById("virtual_count").value'), '4');
  await evaluate('document.getElementById("reset-settings").click()');
  assert.equal(await evaluate('document.getElementById("emergency_yield").checked'), true);
  await evaluate('document.getElementById("emergency_yield").click()');
  assert.equal(await evaluate('document.getElementById("emergency_yield").checked'), false);
  await evaluate('document.getElementById("reset-settings").click()');
  assert.equal(await evaluate('document.getElementById("emergency_yield").checked'), true);
  for (let traffic = 0; traffic < 3; traffic++) for (let demand = 0; demand < 3; demand++) for (let occupancy = 0; occupancy < 3; occupancy++) {
    const actual = await evaluate(`(() => {
      for (const [key,value] of Object.entries({traffic:${traffic},demand:${demand},occupancy:${occupancy}})) document.querySelector('[data-setting="'+key+'"] [data-level="'+value+'"]').click();
      return [settings.traffic,settings.demand,settings.occupancy];
    })()`);
    assert.deepEqual(actual, [traffic, demand, occupancy]);
  }
  await evaluate('document.getElementById("reset-settings").click()');
  await screenshot('desktop-ready.png');
  await call('Emulation.setDeviceMetricsOverride', {width: 390, height: 844, deviceScaleFactor: 1, mobile: true});
  await delay(500);
  assert.equal(await evaluate('document.documentElement.scrollWidth <= innerWidth'), true, 'mobile has no overflow');
  await screenshot('mobile-ready.png');
  await call('Emulation.setDeviceMetricsOverride', {width: 1440, height: 1200, deviceScaleFactor: 1, mobile: false});
  await evaluate(`(() => {
    document.getElementById('duration').value=3600;
    document.getElementById('duration').dispatchEvent(new Event('input'));
    document.querySelector('[data-setting="traffic"] [data-level="${testedTraffic}"]').click();
    document.querySelector('[data-setting="occupancy"] [data-level="0"]').click();
    document.getElementById('speed').value='300';
    document.getElementById('speed').dispatchEvent(new Event('change'));
  })()`);
  if (testedVirtualCount != null) await evaluate(`(() => {
    document.getElementById('policy').value='shortest_time'; document.getElementById('policy').dispatchEvent(new Event('change'));
    document.getElementById('virtual_count').value=${JSON.stringify(String(testedVirtualCount))}; document.getElementById('virtual_count').dispatchEvent(new Event('input'));
  })()`);
  await delay(500);
  ownedRun = true;
  await evaluate('document.getElementById("start").click()');
  await until(() => evaluate('currentState.status === "error" || (currentState.status === "running" && (currentState.frame?.vehicles?.length || 0) > 0)'), 'real ambulances');
  assert.equal(await evaluate('currentState.status'), 'running', await evaluate('currentState.message'));
  assert.ok(await evaluate('currentState.scenario.prob < 0.001 && currentState.scenario.patients_per_hour < 3'));
  assert.equal(await evaluate('currentState.options.emergency_yield'), true);
  assert.equal(await evaluate('currentState.emergency_yield.reaction_distance_m'), 60);
  assert.equal(await evaluate('document.getElementById("emergency_yield").disabled'), true);
  assert.ok(await evaluate('Array.isArray(currentState.frame.traffic) && Array.isArray(currentState.frame.lanes)'));
  if (testedTraffic === 2) await until(() => evaluate('(currentState.frame?.vehicles?.length || 0) > 0 && (currentState.frame?.traffic?.length || 0) > 0'), 'native nearby traffic', 60000);
  assert.equal(await evaluate('document.getElementById("virtual_count").disabled'), true, 'running settings are locked');
  if (testedVirtualCount != null) {
    assert.equal(await evaluate('currentState.virtual_count'), testedVirtualCount);
    assert.equal(await evaluate('currentState.hospitals.filter(h=>h.virtual).length'), testedVirtualCount);
    assert.equal(await evaluate('currentState.options.virtual_count'), testedVirtualCount);
  }
  const before = await evaluate('currentState.frame.vehicles.map(v=>({id:v.id,x:v.x,y:v.y}))');
  await until(async () => {
    const after = await evaluate('currentState.frame.vehicles.map(v=>({id:v.id,x:v.x,y:v.y}))');
    return after.some(v => before.some(b => b.id === v.id && (b.x !== v.x || b.y !== v.y)));
  }, 'ambulances moved', 60000);
  await evaluate('document.getElementById("pause").click()');
  await until(() => evaluate('currentState.status === "paused"'), 'pause');
  await delay(700);
  const pausedTime = await evaluate('currentState.frame.time');
  await delay(700);
  assert.equal(await evaluate('currentState.frame.time'), pausedTime, 'paused clock is stable');
  await screenshot('desktop-simulation.png');
  await evaluate('document.getElementById("focus-ambulance").click()');
  await evaluate(`document.getElementById('follow-ambulance').value=vehicles[0].id; document.getElementById('follow-ambulance').dispatchEvent(new Event('change'))`);
  assert.equal(await evaluate('view.scale'), 4, 'ambulance close-up');
  assert.equal(await evaluate('trafficVehicles.filter(v=>v.yielding).length'), await evaluate('currentState.frame.yielding_count'));
  await delay(350);
  await screenshot('desktop-emergency-closeup.png');
  assert.equal(await evaluate('view.scale'), 4, 'close-up survives viewport resize during capture');
  await evaluate('document.getElementById("map-reset").click()');
  await evaluate('document.getElementById("pause").click()');
  await until(() => evaluate('currentState.status === "running"'), 'resume');
  await until(() => evaluate(`currentState.frame.time > ${pausedTime}`), 'resumed clock');
  await until(() => evaluate('currentState.status === "error" || currentState.patient_count > 0'), 'live patient outcome', 180000);
  assert.notEqual(await evaluate('currentState.status'), 'error', await evaluate('currentState.message'));
  await until(() => evaluate('document.getElementById("result-run").value === "" && document.querySelectorAll(".patient-detail-button").length > 0'), 'live detail table');
  const live = await evaluate('api("/api/patients")');
  assert.ok(live.rows[0].vehicle_id, 'actual ambulance ID retained');
  assert.ok(live.rows[0].hospital_name, 'destination name retained');
  assert.ok(Math.abs(Object.values(live.rows[0].reward_components).reduce((a,b)=>a+b,0) - live.rows[0].reward) < 1e-7);
  const exported = await evaluate('api("/api/export")');
  assert.ok(exported.patient_outcomes.length > 0, 'JSON includes patient records');
  await screenshot('desktop-live-patient-results.png');
  await evaluate('document.getElementById("stop").click()');
  await until(() => evaluate('currentState.status === "stopped"'), 'stop');
  await until(() => evaluate('!document.getElementById("start").disabled'), 'stopped worker cleanup');
  assert.equal(await evaluate('currentState.frame.active'), 0);
  assert.equal(await evaluate('currentState.frame.yielding_count'), 0);
  assert.equal(await evaluate('document.getElementById("focus-ambulance").disabled'), true);
  // Exercise a real backend transition that changes no SUMO time: validation clears its map.
  await evaluate(`(() => {
    const state = structuredClone(currentState);
    state.status='running'; state.options.mode='training'; state.frame.time=123;
    state.frame.vehicles=[{id:'regression',x:0,y:0,angle:90,ktas:1,speed:0}];
    state.frame.traffic=[{id:'ordinary',x:2,y:2,angle:90,speed:0,yielding:false,yielding_to:[]}];
    state.frame_revision=100000; renderState(state);
    state.frame.vehicles=[]; state.frame.traffic=[]; state.frame_revision++;
    renderState(state);
  })()`);
  assert.equal(await evaluate('vehicles.length + trafficVehicles.length'), 0, 'validation removes previous frame vehicles');
  await evaluate('(async () => renderState(await api("/api/state")))()');
  if (process.env.DASHBOARD_CHECK_EMPTY_TRAINING === '1') {
    await checkEmptyTraining();
  }
  assert.deepEqual(errors, [], 'no browser runtime exceptions');
  await writeFile(path.join(output, 'browser-check.json'), JSON.stringify({
    passed: true, scenarioCombinations: 27, movingAmbulances: true, pauseResumeStop: true,
    mobileWidth: 390, desktopWidth: 1440, savedPatients: true, severityFilter: true,
    patientDialog: true, csvExport: true, livePatientOutcomes: true,
    virtualHospitalControls: true, virtualCount: testedVirtualCount, calibratedHourlyDemand: true,
    emergencyYieldControls: true, nativeTrafficTelemetry: true, ambulanceCloseup: true,
    staleFilterResponseIgnored: true, clearedValidationTraffic: true, learningEstimate: true, runtimeErrors: errors
  }, null, 2));
  console.log('PASS: desktop/mobile, 27 scenario combinations, real moving ambulances, pause/resume/stop, no runtime errors.');
  }
} finally {
  if (ownedRun) try {
    const config = await (await fetch(`${baseUrl}/api/config`)).json();
    const state = await (await fetch(`${baseUrl}/api/state`)).json();
    if (['preparing','running','paused'].includes(state.status)) await fetch(`${baseUrl}/api/control`, {
      method: 'POST', headers: {'Content-Type': 'application/json', 'X-Dashboard-Token': config.token}, body: JSON.stringify({action:'stop'})
    });
  } catch {}
  socket?.close(); chrome.kill();
  await delay(1500);
  // Delete only the unique profile created in the OS temporary directory above.
  const resolvedProfile = path.resolve(profile), tempRoot = path.resolve(tmpdir()) + path.sep;
  if (resolvedProfile.startsWith(tempRoot) && path.basename(resolvedProfile).startsWith('mediroute-browser-')) {
    await rm(resolvedProfile, {recursive: true, force: true, maxRetries: 5}).catch(() => {});
  }
}
