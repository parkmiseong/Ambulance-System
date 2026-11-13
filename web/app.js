'use strict';

const $ = (id) => document.getElementById(id);
const icon = (name) => `<svg aria-hidden="true"><use href="#i-${name}"/></svg>`;
const escapeHTML = (value) => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[c]));
const fmt = (value, digits = 0) => Number(value || 0).toLocaleString('ko-KR', {maximumFractionDigits: digits, minimumFractionDigits: digits});
const timeLabel = (seconds) => `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`;
const severityColor = ktas => ['#bd5967', '#cd8a47', '#b5a04c', '#4c9c91', '#668eb8'][Number(ktas)-1] || '#829787';
const settings = {traffic: 1, demand: 1, occupancy: 1, mode: 'simulation'};
const activeStatuses = new Set(['preparing', 'running', 'paused', 'stopping']);
let token, currentState = {status: 'idle'}, network, rewardPoints = [], lastRun, lastEpisode, lastChartTime = -1, pollBusy = false;
let toastTimer, lastHospitalRender = -Infinity;
const modelMetadata = new Map();
let requestedVirtualCount = 2, virtualCountEdited = false, hospitalControls = null;
const virtualTypeKeys = ['local_center', 'local_institution', 'reported_er'];
const virtualTypeLabels = ['권역응급의료센터', '지역응급의료센터', '지역응급의료기관', '응급실운영신고기관'];
let demandProfile = null, admissionControls = null, congestionProfile = null;
const settingsCollapsedKey = 'mediroute.settingsCollapsed';

function setSettingsCollapsed(collapsed, persist = true) {
  const toggle = $('toggle-settings');
  $('settings-options').hidden = collapsed;
  toggle.setAttribute('aria-expanded', String(!collapsed));
  toggle.setAttribute('aria-label', `실행 설정 ${collapsed ? '펼치기' : '접기'}`);
  $('settings-toggle-hint').textContent = collapsed ? '펼치기' : '접기';
  if (persist) {
    try { localStorage.setItem(settingsCollapsedKey, String(collapsed)); } catch {}
  }
}
$('toggle-settings').addEventListener('click', () => setSettingsCollapsed(!$('settings-options').hidden));
try { setSettingsCollapsed(localStorage.getItem(settingsCollapsedKey) === 'true', false); } catch {}

function validateSetting(input) {
  if (input.checkValidity()) return true;
  setSettingsCollapsed(false);
  const advanced = input.closest('details');
  if (advanced) advanced.open = true;
  input.reportValidity();
  input.focus();
  return false;
}

function toast(message) {
  $('toast').textContent = message;
  $('toast').hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { $('toast').hidden = true; }, 6000);
}

async function api(path, body) {
  const response = await fetch(path, body === undefined ? {cache: 'no-store'} : {
    method: 'POST', headers: {'Content-Type': 'application/json', 'X-Dashboard-Token': token}, body: JSON.stringify(body)
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || '서버에 연결할 수 없습니다.');
  return data;
}

function updateMode() {
  const training = settings.mode === 'training';
  document.querySelectorAll('[data-mode]').forEach(button => {
    const selected = button.dataset.mode === settings.mode;
    button.classList.toggle('selected', selected);
    button.setAttribute('aria-pressed', selected);
  });
  $('mode-description').textContent = training ? '새 DQN 모델을 학습하고 별도 시드로 성능을 검증합니다.' : '선택한 정책으로 환자의 이송 과정을 확인합니다.';
  $('policy-field').hidden = training;
  $('model-field').hidden = training || $('policy').value !== 'dqn';
  $('episodes').disabled = !training || activeStatuses.has(currentState.status);
  $('start').innerHTML = icon(training ? 'brain' : 'play') + `<span>${training ? '학습' : '시뮬레이션'} 시작</span>`;
  updateVirtualCount();
  updateDemandEstimate();
  updateAdmissionSettings();
}

function updateAdmissionSettings() {
  $('hospital_rejection').disabled = activeStatuses.has(currentState.status) || admissionControls?.stage !== 'after_arrival';
  $('admission-note').textContent = admissionControls?.stage === 'after_arrival'
    ? `사전 수용 확인 후 출발 · 병원 도착 후 거절 1회당 ${fmt(admissionControls.rejection_penalty)}점 · 같은 구급차로 재이송. 사전 수용 불가는 벌점이 없습니다.`
    : '거절 패널티를 적용하려면 서버를 재시작한 뒤 화면을 새로고침하세요.';
}

function updateDemandInputs() {
  const mode = $('demand_mode').value, active = activeStatuses.has(currentState.status);
  for (const option of $('demand_mode').options) option.disabled = option.value !== 'preset' && !demandProfile?.input_modes?.includes(option.value);
  $('custom-rate-field').hidden = mode !== 'rate';
  $('custom-count-field').hidden = mode !== 'count';
  $('demand_mode').disabled = active;
  $('custom_patients_per_hour').disabled = active || mode !== 'rate';
  $('patient_count').disabled = active || mode !== 'count';
  const seconds = Number($('duration').value);
  $('patient_count').max = Number.isInteger(seconds) && seconds >= 30 ? seconds : 86400;
  $('custom-demand-note').textContent = mode === 'count'
    ? `매회 발생 기간에 정확히 지정한 인원을 무작위 시각에 배치합니다. 0~${fmt(Number($('patient_count').max))}명 · 초당 최대 1명.`
    : mode === 'rate' ? '시뮬레이션 1시간당 기대 인원입니다. 실제 발생 수는 시드에 따라 달라집니다. 0~3,600명/시간.'
    : '3단계를 누르면 강남 기준으로 돌아갑니다.';
}

function updateDemandEstimate() {
  updateDemandInputs();
  if (!demandProfile) {
    $('demand-estimate').textContent = token ? '새 환자 발생률을 적용하려면 서버를 재시작해 주세요.' : '시간당 환자 수 기준을 불러오고 있습니다.';
    return;
  }
  const mode = $('demand_mode').value;
  const duration = Number($('duration').value);
  const count = Number($('patient_count').value);
  const enteredRate = Number($('custom_patients_per_hour').value);
  const validDuration = Number.isFinite(duration) && duration > 0;
  const validCount = $('patient_count').value !== '' && Number.isInteger(count) && count >= 0 && count <= duration;
  const validRate = $('custom_patients_per_hour').value !== '' && Number.isFinite(enteredRate) && enteredRate >= 0 && enteredRate <= 3600;
  const rate = mode === 'rate' ? (validRate ? enteredRate : null) : mode === 'count' ? (validDuration && validCount ? count * 3600 / duration : null) : demandProfile.patients_per_hour[settings.demand];
  const expected = mode === 'count' ? (validCount ? count : null) : validDuration && rate != null ? rate * duration / 3600 : null;
  document.querySelectorAll('[data-setting="demand"] button').forEach(button => {
    button.querySelector('small').textContent = `${fmt(demandProfile.patients_per_hour[Number(button.dataset.level)], 2)}명/h`;
    const selected = mode === 'preset' && Number(button.dataset.level) === settings.demand;
    button.classList.toggle('selected', selected); button.setAttribute('aria-pressed', selected);
  });
  const increase = Math.round((demandProfile.scenario_multipliers[settings.demand] - 1) * 100);
  $('demand-estimate').textContent = mode === 'count'
    ? `회차당 총 ${expected == null ? '—' : fmt(expected)}명 · 기간 내 정확히 지정 인원 발생${rate == null ? '' : ` · 시간당 ${fmt(rate, 2)}명에 해당`}`
    : `시간당 ${rate == null ? '—' : fmt(rate, 2)}명${mode === 'preset' ? (increase ? ` (+${increase}%)` : ' (기준 수준)') : ' (직접 입력)'} · 선택 기간의 기대 인원 ${expected == null ? '—' : fmt(expected, 2)}명`;
  $('demand-reference-note').textContent = mode === 'preset'
    ? `${demandProfile.reference_year}년 강남 119 구급활동 ${demandProfile.reference_value_per_day}건/일을 환자 수요의 대리 기준으로 사용합니다. 발생 수는 확률적으로 달라지며 재생 배속은 발생률을 바꾸지 않습니다.`
    : mode === 'count' ? '사용자 지정 인원입니다. 학습은 매 에피소드에 같은 수를 적용하며, 별도 검증 환자는 학습 인원에 포함하지 않습니다.' : '사용자 지정 발생률입니다. 강남 기준 배율을 추가로 곱하지 않으며 재생 배속은 발생률을 바꾸지 않습니다.';
  $('demand-reference-link').textContent = `${mode === 'preset' ? '' : '비교 참고 · '}${demandProfile.source_title} p.${demandProfile.source_page} ↗`;
  $('demand-reference-link').href = demandProfile.source_url;
  $('learning-estimate').hidden = settings.mode !== 'training';
  if (settings.mode === 'training') {
    const count = expected == null ? null : expected * Number($('episodes').value);
    $('learning-estimate').textContent = `전체 학습에서 ${mode === 'count' ? '총' : '약'} ${count == null ? '—' : fmt(count, mode === 'count' ? 0 : 1)}명 ${mode === 'count' ? '발생' : '발생 예상'}. 첫 가중치 업데이트에는 기본 32개 경험이 필요합니다. 환자 수가 적으면 에피소드를 늘리세요.`;
  }
}
$('duration').addEventListener('input', updateDemandEstimate);
$('episodes').addEventListener('input', updateDemandEstimate);
$('demand_mode').addEventListener('change', updateDemandEstimate);
$('custom_patients_per_hour').addEventListener('input', updateDemandEstimate);
$('patient_count').addEventListener('input', updateDemandEstimate);

function virtualMixWeights() {
  return Object.fromEntries(virtualTypeKeys.map(key => [key, Number($(`virtual_ratio_${key}`).value)]));
}
function virtualMixPlan(count) {
  const inputs = virtualTypeKeys.map(key => $(`virtual_ratio_${key}`));
  inputs.forEach(input => input.setCustomValidity(''));
  if (inputs.some(input => input.value === '' || !Number.isInteger(Number(input.value)) || Number(input.value) < 0 || Number(input.value) > 100)) return null;
  const weights = inputs.map(input => Number(input.value)), total = weights.reduce((a,b)=>a+b,0);
  if (count > 2 && total === 0) {
    inputs[0].setCustomValidity('추가 병원의 유형 비율 중 하나 이상은 0보다 커야 합니다.'); return null;
  }
  if (!Number.isInteger(count) || count < 0 || count > 10 || count === 1) return null;
  const remaining = Math.max(0,count-2), additional = weights.map(w => total ? Math.floor(remaining*w/total) : 0);
  const ranked = weights.map((w,i)=>({i,remainder:total ? (remaining*w)%total : 0})).sort((a,b)=>b.remainder-a.remainder || a.i-b.i);
  const extra = remaining-additional.reduce((a,b)=>a+b,0);
  for (const item of ranked.slice(0,extra)) additional[item.i]++;
  return [Math.min(2,count), ...additional];
}
function virtualMixLabel(counts) {
  return ['권역','지역센터','지역기관','신고기관'].map((label,i)=>`${label} ${counts[i]}개`).join(' · ');
}
function updateVirtualCount() {
  const input = $('virtual_count');
  const fromModel = settings.mode === 'simulation' && $('policy').value === 'dqn';
  const running = activeStatuses.has(currentState.status);
  const model = modelMetadata.get($('model').value);
  if (fromModel && !virtualCountEdited && Number.isInteger(model?.virtual_count)) {
    requestedVirtualCount = model.virtual_count === 1 ? 2 : model.virtual_count;
    input.value = requestedVirtualCount;
  }
  const count = input.value === '' ? NaN : Number(input.value);
  input.setCustomValidity(count === 1 ? '권역 2개를 기본 포함하므로 가상 병원은 0개 또는 2~10개로 설정하세요.' : '');
  const plan = virtualMixPlan(count);
  const savedTypes = model?.virtual_hospital_types;
  const savedCounts = savedTypes && virtualTypeLabels.map(type => savedTypes.filter(value=>value===type).length);
  const changedCount = fromModel && model?.virtual_count != null && count !== model.virtual_count;
  const changed = changedCount || (fromModel && plan && savedCounts && plan.some((value,i)=>value!==savedCounts[i]));
  $('virtual-count-note').textContent = fromModel && !hospitalControls?.simulation_editable
    ? '시뮬레이션 병원 수 변경을 적용하려면 서버를 재시작하세요.'
    : changedCount && model?.hospital_resize_supported === false
      ? '이전 v3 모델은 병원 수가 고정됩니다. 변경하려면 v4 이상 모델이나 다른 정책을 선택하세요.'
      : fromModel && model?.virtual_count != null
        ? `저장 모델: 가상 ${model.virtual_count}개 · 현재 요청 ${input.value || '—'}개. ${changed ? '가중치를 유지하고 병원 수·유형을 바꿔 평가합니다.' : '권역 2개 기본 · 선택한 유형 비율로 평가합니다.'}`
        : '0개는 실제 병원만 사용 · 생성 시 2~10개, 권역 2개 기본 포함';
  input.disabled = running || (fromModel && !hospitalControls?.simulation_editable);
  $('virtual-minus').disabled = input.disabled || Number(input.value) <= 0;
  $('virtual-plus').disabled = input.disabled || Number(input.value) >= 10;
  virtualTypeKeys.forEach(key => { $(`virtual_ratio_${key}`).disabled = running || !hospitalControls?.virtual_mix; });
  $('virtual-mix-summary').textContent = !hospitalControls?.virtual_mix
    ? '유형 비율을 적용하려면 서버를 재시작하세요.'
    : !plan ? '병원 수는 0 또는 2~10개, 비율은 0~100의 정수로 입력하세요. 추가 병원이 있으면 비율 합계가 0보다 커야 합니다.'
    : count === 0 ? '가상 병원을 생성하지 않습니다.' : `생성 예정: ${virtualMixLabel(plan)}`;
}
virtualTypeKeys.forEach(key => $(`virtual_ratio_${key}`).addEventListener('input', updateVirtualCount));

$('virtual_count').addEventListener('input', () => {
  virtualCountEdited = true;
  const value = Number($('virtual_count').value);
  if ($('virtual_count').value !== '' && Number.isInteger(value) && value >= 0 && value <= 10) requestedVirtualCount = value;
  updateVirtualCount();
});
for (const [id, delta] of [['virtual-minus', -1], ['virtual-plus', 1]]) $(id).addEventListener('click', () => {
  virtualCountEdited = true;
  requestedVirtualCount = Math.max(0, Math.min(10, requestedVirtualCount + delta));
  if (requestedVirtualCount === 1) requestedVirtualCount = delta > 0 ? 2 : 0;
  $('virtual_count').value = requestedVirtualCount;
  updateVirtualCount();
});
$('model').addEventListener('change', updateVirtualCount);

document.querySelectorAll('[data-mode]').forEach(button => button.addEventListener('click', () => {
  settings.mode = button.dataset.mode;
  updateMode();
}));
document.querySelectorAll('.segments button').forEach(button => button.addEventListener('click', () => {
  settings[button.parentElement.dataset.setting] = Number(button.dataset.level);
  if (button.parentElement.dataset.setting === 'demand') $('demand_mode').value = 'preset';
  button.parentElement.querySelectorAll('button').forEach(sibling => {
    sibling.classList.toggle('selected', sibling === button);
    sibling.setAttribute('aria-pressed', sibling === button);
  });
  updateDemandEstimate();
}));
$('policy').addEventListener('change', updateMode);
$('reset-settings').addEventListener('click', () => {
  setSettingsCollapsed(false);
  settings.mode = 'simulation';
  document.querySelectorAll('.segments').forEach(group => group.querySelector('[data-level="1"]').click());
  requestedVirtualCount = 2;
  virtualCountEdited = false;
  $('emergency_yield').checked = true;
  $('hospital_rejection').checked = true;
  $('demand_mode').value = 'preset';
  $('custom_patients_per_hour').value = 10;
  $('patient_count').value = 10;
  virtualTypeKeys.forEach((key,i) => { $(`virtual_ratio_${key}`).value = [5,4,1][i]; });
  for (const [id, value] of Object.entries({duration: 3600, ambulances: 10, episodes: 3, seed: 42, policy: 'dqn', virtual_count: 2})) $(id).value = value;
  updateMode();
});

async function refreshModels() {
  const config = await api('/api/config');
  token = config.token;
  demandProfile = config.demand_profile || null;
  hospitalControls = config.hospital_controls || null;
  admissionControls = config.hospital_admission || null;
  congestionProfile = config.hospital_congestion || null;
  $('congestion-setting-note').textContent = congestionProfile
    ? '병원별 배정 직전 점유율 50% 미만은 가점, 초과는 감점입니다. 70%를 넘으면 과밀 감점을 추가합니다.'
    : '새 병원별 혼잡도 보상을 적용하려면 서버를 재시작한 뒤 화면을 새로고침하세요.';
  modelMetadata.clear();
  config.models.forEach(model => modelMetadata.set(model.path, model));
  const previous = $('model').value;
  $('model').replaceChildren(...config.models.map((model, i) => {
    const option = document.createElement('option');
    option.value = model.path;
    option.textContent = `${i === 0 ? '최신 · ' : ''}${model.group} / ${model.name}`;
    option.title = model.path;
    return option;
  }));
  if (config.models.some(model => model.path === previous)) $('model').value = previous;
  if (!config.models.length) {
    $('model').innerHTML = '<option value="">저장된 best 모델 없음</option>';
    if ($('policy').value === 'dqn') $('policy').value = 'shortest_time';
  }
  updateMode();
}
$('refresh-models').addEventListener('click', () => refreshModels().catch(error => toast(error.message)));

$('start').addEventListener('click', async () => {
  if (!congestionProfile) { toast('새 병원별 혼잡도 보상을 적용하려면 서버를 재시작한 뒤 화면을 새로고침하세요.'); return; }
  const options = {...settings, policy: $('policy').value, model: $('model').value,
    emergency_yield: $('emergency_yield').checked, demand_mode: $('demand_mode').value,
    hospital_rejection: $('hospital_rejection').checked};
  if (options.hospital_rejection && admissionControls?.stage !== 'after_arrival') {
    toast('병원 거절 패널티를 적용하려면 서버를 재시작하고 화면을 새로고침하세요.'); return;
  }
  if (Number($('virtual_count').value) > 2 && !hospitalControls?.virtual_mix) {
    toast('가상 병원 유형 비율을 적용하려면 서버를 재시작하고 화면을 새로고침하세요.'); return;
  }
  for (const key of virtualTypeKeys) {
    const input = $(`virtual_ratio_${key}`);
    if (!validateSetting(input)) return;
  }
  options.virtual_mix = virtualMixWeights();
  if (options.demand_mode !== 'preset' && !demandProfile?.input_modes?.includes(options.demand_mode)) {
    toast('환자 수 직접 입력을 적용하려면 서버를 재시작한 뒤 화면을 새로고침하세요.'); return;
  }
  const demandInput = options.demand_mode === 'rate' ? 'custom_patients_per_hour' : options.demand_mode === 'count' ? 'patient_count' : null;
  if (demandInput) {
    if (!validateSetting($(demandInput))) return;
    options[demandInput] = Number($(demandInput).value);
  }
  for (const name of ['duration', 'ambulances', 'episodes', 'seed', 'virtual_count']) {
    if (name === 'virtual_count' && $(name).disabled) continue;
    if (!validateSetting($(name))) return;
    options[name] = Number($(name).value);
  }
  $('start').disabled = true;
  try { renderState(await api('/api/start', options)); }
  catch (error) { toast(error.message); $('start').disabled = false; }
});
$('pause').addEventListener('click', async () => {
  try { renderState(await api('/api/control', {action: currentState.status === 'paused' ? 'resume' : 'pause'})); }
  catch (error) { toast(error.message); }
});
$('stop').addEventListener('click', async () => {
  try { renderState(await api('/api/control', {action: 'stop'})); }
  catch (error) { toast(error.message); }
});
$('speed').addEventListener('change', async () => {
  try { renderState(await api('/api/control', {action: 'speed', value: Number($('speed').value)})); }
  catch (error) { toast(error.message); }
});

const canvas = $('map'), ctx = canvas.getContext('2d');
const view = {x: 0, y: 0, scale: .07, width: 0, height: 0, baseScale: .07};
const roadPaths = [new Path2D(), new Path2D()];
const baseMap = document.createElement('canvas');
let baseMapKey = '';
let baseMapAnchor = {x: 0, y: 0};
let mapHospitals = [], vehicles = [], trafficVehicles = [], nearbyLanes = [], mapPatients = [], selectedHospital, pointer, moved = false;
let followedVehicleId = null, followOptionsKey = '', followMessage = '';
const trails = new Map();
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const project = (x, y) => [view.width / 2 + (x - view.x) * view.scale, view.height / 2 - (y - view.y) * view.scale];
const ambulanceName = vehicle => `구급차 ${String(vehicle.id).match(/(\d+)$/)?.[1] || vehicle.id}`;
function interpolatedPosition(vehicle, now) {
  const fraction = reducedMotion ? 1 : Math.max(0, Math.min(1, (now - vehicle.at) / 220));
  return {x: vehicle.fromX + (vehicle.x - vehicle.fromX) * fraction,
    y: vehicle.fromY + (vehicle.y - vehicle.fromY) * fraction};
}
function stopFollowing(message = '') {
  followedVehicleId = null; followMessage = message;
  updateFollowControls();
}
function updateFollowControls() {
  const followed = vehicles.find(vehicle => vehicle.id === followedVehicleId);
  if (followedVehicleId && !followed) {
    followedVehicleId = null;
    followMessage = '선택한 구급차의 운행이 종료되었습니다. 다른 차량을 선택하세요.';
  }
  const entries = [...vehicles].sort((a,b) => a.id.localeCompare(b.id, 'en', {numeric:true}));
  const key = JSON.stringify(entries.map(v => [v.id,v.ktas,v.patient_id]));
  if (key !== followOptionsKey) {
    $('follow-ambulance').replaceChildren(new Option(entries.length ? '구급차를 선택하세요' : '운행 중인 구급차 없음', ''),
      ...entries.map(v => new Option(`${ambulanceName(v)} · KTAS ${v.ktas} · 환자 ${v.patient_id || '—'}`, v.id)));
    followOptionsKey = key;
  }
  $('follow-ambulance').value = followedVehicleId || '';
  $('follow-ambulance').disabled = !vehicles.length;
  $('follow-cancel').disabled = !followedVehicleId;
  $('focus-ambulance').disabled = !vehicles.length;
  $('focus-ambulance').classList.toggle('tracking', Boolean(followedVehicleId));
  $('focus-ambulance').textContent = followed ? `${ambulanceName(followed)} 추적 중` : '구급차 포커싱';
  $('follow-status').textContent = followed ? `${ambulanceName(followed)}를 따라갑니다. 지도 드래그 또는 추적 해제로 자유 탐색합니다.`
    : followMessage || '차량을 선택하면 카메라가 따라갑니다. 확대·축소도 가능합니다.';
}

function resetMap() {
  if (!network) return;
  const hospitals = mapHospitals.length ? mapHospitals : network.hospitals;
  const xs = hospitals.map(h => h.sumo_x), ys = hospitals.map(h => h.sumo_y);
  const left = Math.min(...xs) - 1400, right = Math.max(...xs) + 1400;
  const bottom = Math.min(...ys) - 1000, top = Math.max(...ys) + 1200;
  view.x = (left + right) / 2; view.y = (bottom + top) / 2;
  view.baseScale = Math.min((view.width - 50) / (right - left), (view.height - 70) / (top - bottom));
  view.scale = view.baseScale;
}

function resizeMap() {
  const box = canvas.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  const first = !view.width;
  const fitted = Math.abs(view.scale - view.baseScale) < .000001;
  const previous = {x: view.x, y: view.y, scale: view.scale};
  view.width = box.width; view.height = box.height;
  canvas.width = Math.round(box.width * dpr); canvas.height = Math.round(box.height * dpr);
  if (first || network) {
    resetMap();
    if (!first && !fitted) Object.assign(view, previous);
  }
}
new ResizeObserver(resizeMap).observe($('map-container'));
$('zoom-in').addEventListener('click', () => zoom(1.3));
$('zoom-out').addEventListener('click', () => zoom(1 / 1.3));
$('map-reset').addEventListener('click', () => { stopFollowing('전체 지도로 전환했습니다.'); resetMap(); });
$('focus-ambulance').addEventListener('click', () => {
  $('ambulance-focus-panel').hidden = !$('ambulance-focus-panel').hidden;
  $('focus-ambulance').setAttribute('aria-expanded', String(!$('ambulance-focus-panel').hidden));
  if (!$('ambulance-focus-panel').hidden) $('follow-ambulance').focus();
});
$('close-focus-panel').addEventListener('click', () => {
  $('ambulance-focus-panel').hidden = true;
  $('focus-ambulance').setAttribute('aria-expanded', 'false');
});
$('follow-ambulance').addEventListener('change', () => {
  const vehicle = vehicles.find(v => v.id === $('follow-ambulance').value);
  if (!vehicle) { stopFollowing('추적을 해제했습니다.'); return; }
  followedVehicleId = vehicle.id; followMessage = '';
  view.x = vehicle.x; view.y = vehicle.y; view.scale = 4;
  $('map-tooltip').hidden = true;
  updateFollowControls();
});
$('follow-cancel').addEventListener('click', () => stopFollowing('추적을 해제했습니다. 지도를 자유롭게 이동하세요.'));
function zoom(factor, px = view.width / 2, py = view.height / 2) {
  if (followedVehicleId) { px = view.width / 2; py = view.height / 2; }
  const worldX = view.x + (px - view.width / 2) / view.scale;
  const worldY = view.y - (py - view.height / 2) / view.scale;
  view.scale = Math.min(6, Math.max(view.baseScale * .4, view.scale * factor));
  view.x = worldX - (px - view.width / 2) / view.scale;
  view.y = worldY + (py - view.height / 2) / view.scale;
}
canvas.addEventListener('wheel', event => {
  event.preventDefault();
  const rect = canvas.getBoundingClientRect();
  zoom(Math.exp(-event.deltaY * .0015), event.clientX - rect.left, event.clientY - rect.top);
}, {passive: false});
canvas.addEventListener('pointerdown', event => {
  pointer = {id: event.pointerId, x: event.clientX, y: event.clientY, worldX: view.x, worldY: view.y};
  moved = false;
  canvas.setPointerCapture(event.pointerId);
});
canvas.addEventListener('pointermove', event => {
  if (pointer) {
    const dx = event.clientX - pointer.x, dy = event.clientY - pointer.y;
    if (Math.abs(dx) + Math.abs(dy) > 4) {
      if (!moved && followedVehicleId) stopFollowing('지도 이동으로 추적을 해제했습니다.');
      moved = true;
    }
    view.x = pointer.worldX - dx / view.scale;
    view.y = pointer.worldY + dy / view.scale;
    $('map-tooltip').hidden = true;
  }
});
canvas.addEventListener('pointerup', event => {
  pointer = null;
  if (moved) return;
  const rect = canvas.getBoundingClientRect(), px = event.clientX - rect.left, py = event.clientY - rect.top;
  const vehicle = vehicles.find(v => { const [x, y] = project(v.x, v.y); return Math.hypot(x-px, y-py) < 20; });
  if (vehicle) {
    selectedHospital = mapHospitals.find(h => h.id === vehicle.hospital_id);
    mapTooltip(px, py, `<b>환자 ${escapeHTML(vehicle.patient_id || '—')} · KTAS ${Number(vehicle.ktas)}</b><small>이송 임무 · ${escapeHTML(vehicle.id)}</small><br>목적지 ${escapeHTML(vehicle.hospital_name || selectedHospital?.name || vehicle.hospital_id)}<br>운행 ${fmt(vehicle.elapsed_seconds || 0)}초 · ${fmt(vehicle.speed*3.6, 1)} km/h${vehicle.rejection_count ? `<br>도착 후 거절 ${fmt(vehicle.rejection_count)}회 · ${vehicle.transport_stage === 'awaiting_hospital' ? '병원에서 수용·재이송 대기' : '다른 병원으로 재이송 중'}` : ''}`);
    return;
  }
  const car = trafficVehicles.find(v => { const [x,y] = project(v.x,v.y); return Math.hypot(x-px,y-py) < Math.max(6, 2*view.scale); });
  if (car) {
    mapTooltip(px, py, `<b>${car.yielding ? '구급차에 양보 중' : '주변 차량'}</b><small>${escapeHTML(car.id)}</small><br>${fmt(car.speed*3.6, 1)} km/h${car.yielding ? `<br>양보 대상 ${car.yielding_to.map(escapeHTML).join(', ')}` : ''}`);
    return;
  }
  selectedHospital = mapHospitals.find(h => { const [x, y] = project(h.sumo_x, h.sumo_y); return Math.hypot(x - px, y - py) < 20; });
  if (selectedHospital) { showHospitalTooltip(selectedHospital); return; }
  const patient = mapPatients.find(p => { const [x, y] = project(p.x, p.y); return Math.hypot(x-px, y-py) < 9; });
  if (patient) { mapTooltip(px, py, `<b>환자 ${escapeHTML(patient.id)} · KTAS ${Number(patient.ktas)}</b><small>병원 배정을 기다리고 있습니다.</small>`); return; }
  const center = $('show-centers').checked && network?.centers?.find(c => { const [x,y] = project(c.x,c.y); return Math.hypot(x-px,y-py) < 15; });
  if (center) { mapTooltip(px, py, `<b>${escapeHTML(center.name)}</b><small>119/소방 참고 위치 · 실제 출발지는 환자 위치</small>`); return; }
  $('map-tooltip').hidden = true;
});
canvas.addEventListener('pointercancel', () => { pointer = null; });

function showHospitalTooltip(hospital) {
  if (!hospital) { $('map-tooltip').hidden = true; return; }
  const [x, y] = project(hospital.sumo_x, hospital.sumo_y);
  mapTooltip(x, y, `<b>${escapeHTML(hospital.name)}</b><small>${escapeHTML(hospital.type)}${hospital.virtual ? ' · 가상 병원' : ''}</small><br>점유율 ${hospital.occupancy == null ? '실행 대기' : fmt(hospital.occupancy * 100) + '%'} · 예약 ${hospital.reserved_count || 0}명`);
}
function mapTooltip(x, y, html) {
  $('map-tooltip').innerHTML = html;
  $('map-tooltip').style.left = `${Math.max(10, Math.min(view.width - 245, x + 20))}px`;
  $('map-tooltip').style.top = `${Math.max(65, Math.min(view.height - 100, y - 50))}px`;
  $('map-tooltip').hidden = false;
}

function roundedRect(x, y, w, h, radius) { ctx.beginPath(); ctx.roundRect(x, y, w, h, radius); }
function drawBaseMap(dpr) {
  // Pan a padded road bitmap between refreshes so following stays smooth.
  const padding = 256, width = view.width + padding*2, height = view.height + padding*2;
  const key = [view.scale, view.width, view.height, dpr].join(',');
  let dx = (baseMapAnchor.x-view.x)*view.scale, dy = (view.y-baseMapAnchor.y)*view.scale;
  if (key !== baseMapKey || Math.abs(dx) > padding-32 || Math.abs(dy) > padding-32) {
    baseMap.width = Math.round(width*dpr); baseMap.height = Math.round(height*dpr);
    const background = baseMap.getContext('2d');
    background.setTransform(dpr,0,0,dpr,0,0);
    background.fillStyle = '#e9eee5'; background.fillRect(0,0,width,height);
    background.save(); background.translate(width/2,height/2);
    background.scale(view.scale,-view.scale); background.translate(-view.x,-view.y);
    background.lineJoin = 'round'; background.lineCap = 'round';
    background.strokeStyle = '#dde4d9'; background.lineWidth = Math.max(9,2.7/view.scale); background.stroke(roadPaths[0]);
    background.strokeStyle = '#f9faf4'; background.lineWidth = Math.max(5,1.5/view.scale); background.stroke(roadPaths[0]);
    background.strokeStyle = '#d5decf'; background.lineWidth = Math.max(19,5/view.scale); background.stroke(roadPaths[1]);
    background.strokeStyle = '#fffef6'; background.lineWidth = Math.max(13,3.7/view.scale); background.stroke(roadPaths[1]);
    background.restore(); background.textAlign = 'center';
    for (const [name,x,y] of network.districts) {
      const [px,py] = project(x,y);
      background.font = '600 14px "Malgun Gothic", sans-serif'; background.fillStyle = '#9aaa92';
      background.fillText(name,px+padding,py+padding);
    }
    if (view.scale > .055) {
      const used = [];
      for (const [name,x,y] of network.labels) {
        const [px,py] = project(x,y);
        if (px < -padding+50 || px > view.width+padding-55 || py < -padding+50 || py > view.height+padding-35 || used.some(([a,b]) => Math.hypot(px-a,py-b) < 100)) continue;
        used.push([px,py]); background.font = '8px "Malgun Gothic", sans-serif'; background.fillStyle = '#acb8a4';
        background.fillText(name,px+padding,py+padding);
      }
    }
    baseMapAnchor = {x:view.x,y:view.y}; baseMapKey = key; dx = 0; dy = 0;
  }
  ctx.drawImage(baseMap,dx-padding,dy-padding,width,height);
}
function drawMap(now) {
  const followed = vehicles.find(vehicle => vehicle.id === followedVehicleId);
  if (followed) {
    const position = interpolatedPosition(followed, now);
    view.x = position.x; view.y = position.y;
  }
  const dpr = window.devicePixelRatio || 1;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, view.width, view.height);
  ctx.fillStyle = '#e9eee5'; ctx.fillRect(0, 0, view.width, view.height);
  if (network) {
    drawBaseMap(dpr);
    if (view.scale > .3) {
      ctx.save(); ctx.strokeStyle = '#aebca5'; ctx.lineWidth = .65; ctx.setLineDash([4,5]);
      for (const lane of nearbyLanes) {
        ctx.beginPath();
        lane.shape.forEach(([x,y], i) => {const [px,py] = project(x,y); if (i) ctx.lineTo(px,py); else ctx.moveTo(px,py);});
        ctx.stroke();
      }
      ctx.restore();
    }
    for (const car of trafficVehicles) {
      const fraction = reducedMotion ? 1 : Math.min(1, (now-car.at)/220);
      const [x,y] = project(car.fromX+(car.x-car.fromX)*fraction, car.fromY+(car.y-car.fromY)*fraction);
      if (x < -40 || x > view.width+40 || y < -40 || y > view.height+40) continue;
      const length = Math.max(4.5, (car.length || 5)*view.scale), width = Math.max(2.5, (car.width || 1.8)*view.scale);
      ctx.save(); ctx.translate(x,y); ctx.rotate((car.angle-90)*Math.PI/180);
      roundedRect(-length, -width/2, length, width, Math.min(2,width/3));
      ctx.fillStyle = car.yielding ? '#d49a32' : '#89968c'; ctx.fill();
      if (car.yielding) {ctx.strokeStyle = '#875c16'; ctx.lineWidth = 1; ctx.stroke();}
      ctx.restore();
    }
    if ($('show-centers').checked) {
      const placed = [];
      for (const center of network.centers || []) {
        const [x,y] = project(center.x, center.y);
        if (x < 0 || x > view.width || y < 0 || y > view.height || placed.some(([a,b]) => Math.hypot(x-a,y-b) < 15)) continue;
        placed.push([x,y]);
        roundedRect(x-11,y-9,22,18,5); ctx.fillStyle = '#fff9e7'; ctx.fill(); ctx.strokeStyle = '#ccab68'; ctx.lineWidth = 1; ctx.stroke();
        ctx.fillStyle = '#b49857'; ctx.font = 'bold 7px "Segoe UI", sans-serif'; ctx.textAlign = 'center'; ctx.fillText('119',x,y+2.5);
      }
    }
    for (const patient of mapPatients) {
      const [x, y] = project(patient.x, patient.y);
      ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2); ctx.fillStyle = severityColor(patient.ktas); ctx.fill();
      ctx.strokeStyle = '#fff8e6'; ctx.lineWidth = 1.5; ctx.stroke();
    }
    for (const hospital of mapHospitals) {
      const [x, y] = project(hospital.sumo_x, hospital.sumo_y);
      if (x < -50 || x > view.width+50 || y < -50 || y > view.height+50) continue;
      if (hospital.id === selectedHospital?.id) { ctx.beginPath(); ctx.arc(x, y, 23, 0, 2*Math.PI); ctx.fillStyle = '#659db620'; ctx.fill(); }
      ctx.save(); ctx.shadowColor = '#4d6c6522'; ctx.shadowBlur = 7; ctx.shadowOffsetY = 2;
      roundedRect(x - 13, y - 14, 26, 27, 7); ctx.fillStyle = '#ffffff'; ctx.fill();
      ctx.shadowColor = 'transparent'; ctx.strokeStyle = hospital.virtual ? '#9cac9d' : '#7895b2';
      if (hospital.virtual) ctx.setLineDash([3, 2]); ctx.lineWidth = 1.5; ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = hospital.virtual ? '#96a594' : '#6e8eaf'; ctx.font = 'bold 16px "Segoe UI", sans-serif'; ctx.textAlign = 'center'; ctx.fillText('H', x, y + 5);
      if (hospital.occupancy != null) {
        ctx.beginPath(); ctx.arc(x+11, y-11, 3.7, 0, 2*Math.PI);
        ctx.fillStyle = occupancyColor(hospital.occupancy); ctx.fill(); ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; ctx.stroke();
      }
      if (view.scale > .085 || hospital.id === selectedHospital?.id) {
        const name = shortName(hospital.name);
        ctx.font = '9px "Malgun Gothic", sans-serif'; ctx.fillStyle = '#6a7e6b';
        ctx.strokeStyle = '#f7f9f0'; ctx.lineWidth = 3; ctx.strokeText(name, x, y+27); ctx.fillText(name, x, y+27);
      }
      ctx.restore();
    }
    for (const vehicle of vehicles) {
      const position = interpolatedPosition(vehicle, now);
      const wx = position.x, wy = position.y;
      const trail = trails.get(vehicle.id) || [];
      if (trail.length > 1) {
        ctx.beginPath(); trail.forEach(([tx, ty], i) => { const [px, py] = project(tx, ty); if(i) ctx.lineTo(px, py); else ctx.moveTo(px, py); });
        ctx.strokeStyle = '#148f7655'; ctx.lineWidth = 2; ctx.stroke();
      }
      const [x, y] = project(wx, wy);
      if (vehicle.id === followedVehicleId) {
        ctx.beginPath(); ctx.arc(x, y, 18, 0, Math.PI*2);
        ctx.fillStyle = '#d8b44e22'; ctx.fill(); ctx.strokeStyle = '#b7a45c'; ctx.lineWidth = 1.5; ctx.stroke();
      }
      ctx.save(); ctx.translate(x, y); ctx.rotate((vehicle.angle - 90) * Math.PI / 180);
      const size = view.scale > .35 ? Math.max(.3, (vehicle.length || 5)*view.scale/26) : 1;
      ctx.scale(size, size); ctx.translate(-13, 0);
      ctx.shadowColor = '#227b5750'; ctx.shadowBlur = 8; ctx.shadowOffsetY = 2;
      roundedRect(-13, -7, 26, 14, 4); ctx.fillStyle = '#138d77'; ctx.fill();
      ctx.shadowColor = 'transparent'; ctx.lineWidth = 2.2; ctx.strokeStyle = severityColor(vehicle.ktas); ctx.stroke();
      ctx.fillStyle = '#dff5e9'; roundedRect(5, -5, 5, 10, 1.5); ctx.fill();
      ctx.fillStyle = '#fff'; ctx.fillRect(-7, -1.3, 8, 2.6); ctx.fillRect(-4.3, -4, 2.6, 8);
      ctx.fillStyle = '#ef8c75'; ctx.fillRect(1, -8.5, 4, 2); ctx.restore();
      if (vehicle.id === followedVehicleId || !$('ambulance-focus-panel').hidden) {
        ctx.font = '600 10px "Malgun Gothic", sans-serif'; ctx.textAlign = 'center';
        ctx.strokeStyle = '#fffef5'; ctx.lineWidth = 3; ctx.strokeText(ambulanceName(vehicle), x, y-22);
        ctx.fillStyle = '#6c7c54'; ctx.fillText(ambulanceName(vehicle), x, y-22);
      }
    }
    const meters = 65 / view.scale;
    $('scale-label').textContent = meters >= 1000 ? `${(meters/1000).toFixed(1)} km` : `${Math.round(meters/10)*10} m`;
  }
  requestAnimationFrame(drawMap);
}
requestAnimationFrame(drawMap);

function shortName(name) {
  return String(name).replace('학교법인가톨릭학원가톨릭대학교', '').replace('학교법인', '').replace('연세대학교의과대학', '').replace('재단법인베스티안재단', '').replace('차의과학대학교', '').replace('재단법인', '');
}
function occupancyColor(value) { return value >= .65 ? '#d88b7d' : value >= .4 ? '#cfac6e' : '#63ab8e'; }
function hospitalCongestionPreview(hospital) {
  const score = hospital.congestion_reward_preview;
  return score == null ? '' : `<small class="hospital-congestion-score ${score > 0 ? 'positive' : ''}" title="현재 혼잡 비율에 따른 보상 항목의 참고 점수입니다. 실제 점수는 배정 직전 비율과 이송 결과에 따라 확정됩니다.">혼잡도 보상 ${score > 0 ? '+' : ''}${fmt(score, 2)}점</small>`;
}
function renderHospitals(hospitals) {
  mapHospitals = hospitals.filter(h => h.sumo_x != null && h.sumo_y != null);
  $('hospital-count').textContent = hospitals.length;
  if (performance.now() - lastHospitalRender < 600) return;
  lastHospitalRender = performance.now();
  $('hospital-list').innerHTML = hospitals.map(h => `<div class="hospital" role="button" tabindex="0" data-id="${escapeHTML(h.id)}" aria-label="${escapeHTML(h.name)} 상세 보기"><span class="hospital-mini">${icon('hospital')}</span><div class="hospital-info"><div class="hospital-name" title="${escapeHTML(h.name)}">${escapeHTML(shortName(h.name))}</div><div class="hospital-type">${escapeHTML(h.type)}${h.virtual ? ' · 가상' : ''}</div>${hospitalCongestionPreview(h)}</div><div class="occupancy"><span class="occupancy-track"><i style="width:${Math.min(100, (h.occupancy || 0)*100)}%;background:${occupancyColor(h.occupancy)}"></i></span><b>${h.occupancy == null ? '—' : fmt(h.occupancy*100)+'%'}</b></div></div>`).join('');
}
function focusHospital(element) {
  stopFollowing('병원 위치로 이동했습니다.');
  const hospital = mapHospitals.find(h => h.id === element.dataset.id);
  if (!hospital) return;
  view.x = hospital.sumo_x; view.y = hospital.sumo_y;
  view.scale = Math.max(view.scale, .15); selectedHospital = hospital;
  showHospitalTooltip(hospital);
}
$('hospital-list').addEventListener('click', event => { const item = event.target.closest('.hospital'); if(item) focusHospital(item); });
$('hospital-list').addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') { const item = event.target.closest('.hospital'); if(item) { event.preventDefault(); focusHospital(item); } } });

function renderChart(points, training) {
  const chart = $('chart');
  const guides = [10, 45, 80].map(y => `<path d="M0 ${y}H320" stroke="#eaf1e9" stroke-dasharray="3 4"/>`).join('');
  $('chart-empty').hidden = points.length > 0;
  if (!points.length) { chart.innerHTML = guides; return; }
  const xs = points.map(p => p[0]), ys = points.map(p => p[1]);
  let low = Math.min(0, ...ys), high = Math.max(0, ...ys);
  if (high === low) { low -= 1; high += 1; }
  const minX = Math.min(...xs), actualMaxX = Math.max(...xs), maxX = Math.max(actualMaxX, minX + 1);
  const coordinates = points.map(([x,y]) => [5+(x-minX)/(maxX-minX)*310, 85-(y-low)/(high-low)*75]);
  const line = coordinates.map(([x,y],i) => `${i?'L':'M'}${x.toFixed(1)},${y.toFixed(1)}`).join(' ');
  const [lastX,lastY] = coordinates.at(-1), firstX = coordinates[0][0];
  chart.innerHTML = `<defs><linearGradient id="area" x1="0" y1="0" x2="0" y2="1"><stop stop-color="#53a58a" stop-opacity=".14"/><stop offset="1" stop-color="#53a58a" stop-opacity="0"/></linearGradient></defs>${guides}<path d="${line} L${lastX},95 L${firstX},95Z" fill="url(#area)" stroke="none"/><path d="${line}" fill="none" stroke="#57a58b" stroke-width="1.8"/><circle cx="${lastX}" cy="${lastY}" r="3" fill="#57a58b" stroke="#fff" stroke-width="2"/>`;
  chart.setAttribute('aria-label', `${training ? '에피소드별 환자당' : '누적'} 보상: ${fmt(ys.at(-1), 2)}점`);
  $('chart-start').textContent = training ? '1회' : `${fmt(minX)}초`;
  $('chart-end').textContent = training ? `${fmt(actualMaxX)}회` : `${fmt(actualMaxX)}초`;
}

function renderReward(state) {
  const f = state.frame || {}, result = state.result, data = result || f;
  const total = data.reward, n = result ? data.generated_patients : data.completed;
  $('reward-version').textContent = `REWARD v${data.reward_version || state.reward_version || 4}`;
  $('total-reward').innerHTML = `${total == null ? '—' : fmt(total, 1)}<small>점</small>`;
  $('avg-reward').textContent = n ? fmt((result ? total : (f.completed_reward ?? total))/n, 2) : '—';
  $('avg-reward').title = result ? '전체 발생 환자당 보상' : '결과가 확정된 환자당 보상';
  $('golden-rate').textContent = n ? `${fmt((data.golden_success || 0) / n * 100, 1)}%` : '—';
  $('golden-rate').title = result ? '전체 발생 환자 대비 골든타임 내 도착 비율' : '현재 결과가 확정된 환자 대비 비율 (실행 중)';
  $('failed-count').textContent = data.failed == null ? '—' : `${fmt(data.failed)}명`;
  $('rejection-count').textContent = `${fmt(result?.rejection_count ?? Math.max(f.rejection_count || 0, state.live_rejection_count || 0))}회`;
  $('rejection-count').title = '병원에 실제 도착한 뒤 수용을 거절당한 횟수. 사전 문의와 반복 상태 확인은 제외합니다.';
  const training = state.options?.mode === 'training';
  $('chart-title').textContent = training ? '에피소드별 환자당 보상' : '누적 보상 추이';
  if (f.time != null && f.time !== lastChartTime) {
    rewardPoints.push([f.time, f.reward || 0]); lastChartTime = f.time;
    if (rewardPoints.length > 400) rewardPoints = rewardPoints.filter((_, i) => i % 2 === 0);
  } else if (!result && rewardPoints.length && rewardPoints.at(-1)[0] === f.time) {
    rewardPoints[rewardPoints.length-1][1] = f.reward || 0;
  }
  if (result && !training && rewardPoints.at(-1)?.[0] !== result.simulation_end_time) rewardPoints.push([result.simulation_end_time, result.reward]);
  renderChart(training ? (state.history || []).map(item => [item.episode, item.result.avg_reward]) : rewardPoints, training);
  const parts = data.reward_components || {};
  const groups = [['골든타임', parts.golden || 0], ['이동·시간 초과', (parts.travel || 0) + (parts.overtime || 0)], ['병원 적합도', parts.mismatch || 0], ['병원별 혼잡도', (parts.occupancy || 0)+(parts.overcrowding || 0)], ['수용 거절', parts.rejection || 0], ['이송 실패', parts.route_failure || 0]];
  const max = Math.max(1, ...groups.map(p => Math.abs(p[1])));
  $('reward-components').innerHTML = groups.map(([name,value]) => `<div class="component-row ${value < 0 ? 'negative' : ''}"><span>${name}</span><span class="component-track"><i style="width:${Math.abs(value)/max*100}%"></i></span><b>${value > 0 ? '+' : ''}${fmt(value, 1)}</b></div>`).join('');
  const learning = training ? ` · 가중치 업데이트 누적 ${fmt(result?.training_updates ?? f.updates ?? 0)}회${(result?.training_updates ?? f.updates ?? 0) === 0 ? ' (아직 가중치가 학습되지 않았습니다)' : ''}` : '';
  const validation = result?.validation_generated_patients === 0 ? ' · 검증 환자 0명: 성능 평가 불가' : result?.validation_avg_reward != null ? ` · 검증 보상 ${fmt(result.validation_avg_reward, 2)}` : '';
  $('result-note').textContent = state.output ? `저장 위치: ${state.output}${learning}${validation}` : '환자의 도착·실패가 확정되면 보상에 반영됩니다.';
}

function renderState(state) {
  currentState = state;
  const active = activeStatuses.has(state.status), f = state.frame || {}, result = state.result;
  if (state.run_id && state.run_id !== lastRun && state.options) {
    const options = state.options;
    settings.mode = options.mode;
    for (const key of ['traffic', 'demand', 'occupancy']) {
      settings[key] = options[key];
      document.querySelectorAll(`[data-setting="${key}"] button`).forEach(button => {
        const selected = Number(button.dataset.level) === options[key];
        button.classList.toggle('selected', selected); button.setAttribute('aria-pressed', selected);
      });
    }
    for (const key of ['duration', 'ambulances', 'episodes', 'seed', 'policy']) $(key).value = options[key];
    $('demand_mode').value = options.demand_mode || 'preset';
    if (options.custom_patients_per_hour != null) $('custom_patients_per_hour').value = options.custom_patients_per_hour;
    if (options.patient_count != null) $('patient_count').value = options.patient_count;
    $('emergency_yield').checked = options.emergency_yield !== false;
    $('hospital_rejection').checked = options.hospital_rejection !== false;
    if (Number.isInteger(options.virtual_count)) {
      requestedVirtualCount = options.virtual_count;
      virtualCountEdited = true;
      $('virtual_count').value = requestedVirtualCount;
    }
    if (options.virtual_mix) virtualTypeKeys.forEach(key => { $(`virtual_ratio_${key}`).value = options.virtual_mix[key]; });
    if (options.model) $('model').value = options.model;
    updateMode();
  }
  if (state.run_id !== lastRun || state.episode !== lastEpisode) {
    rewardPoints = []; lastChartTime = -1; trails.clear(); vehicles = []; trafficVehicles = []; nearbyLanes = []; mapPatients = [];
    lastRun = state.run_id; lastEpisode = state.episode;
    renderState.lastFrame = undefined; renderState.lastFrameRevision = undefined;
    followedVehicleId = null; followMessage = ''; followOptionsKey = '';
  }
  document.querySelectorAll('.mode-tab,.segments button,#policy,#model,#reset-settings,#duration,#ambulances,#episodes,#seed,#refresh-models,#emergency_yield').forEach(element => { element.disabled = active; });
  $('start').disabled = active || state.worker_active || !token || !demandProfile;
  if (!active) $('episodes').disabled = settings.mode !== 'training';
  updateVirtualCount();
  updateDemandInputs();
  updateAdmissionSettings();
  $('pause').disabled = !['running', 'paused'].includes(state.status);
  $('pause').innerHTML = icon(state.status === 'paused' ? 'play' : 'pause');
  $('pause').setAttribute('aria-label', state.status === 'paused' ? '계속 실행' : '일시정지');
  $('stop').disabled = !['preparing', 'running', 'paused'].includes(state.status);
  if (state.speed) $('speed').value = state.speed;
  const statusText = {idle: '준비', preparing: '준비 중', running: '실행 중', paused: '일시정지', stopping: '중지 중', completed: '완료', stopped: '중지', error: '오류'};
  $('map-status').className = `status-badge ${state.status}`;
  $('map-status').innerHTML = `<i></i>${statusText[state.status] || '준비'}`;
  $('run-message').className = `run-message ${state.status === 'error' ? 'error' : ''}`;
  $('run-message').querySelector('span').textContent = !active && state.worker_active ? '직전 실행을 정리하고 있습니다. 잠시 후 다시 시작할 수 있습니다.' : !active && !demandProfile ? '새 발생률 적용을 위해 서버를 재시작한 뒤 화면을 새로고침하세요.' : state.status === 'paused' ? '일시정지 상태입니다. 재생 버튼으로 계속할 수 있습니다.' : state.message;
  $('active-count').textContent = fmt(f.active || 0);
  $('arrived-count').textContent = fmt(result?.arrived ?? f.arrived ?? 0);
  $('waiting-count').textContent = fmt(f.waiting || 0);
  $('yielding-count').textContent = `주변 양보 ${fmt(f.yielding_count || 0)}대`;
  $('yielding-count').title = '현재 이송 중인 구급차 반경 100m 내 표시 차량 기준 (최대 150대)';
  $('ambulance-limit').textContent = state.options?.ambulances || Number($('ambulances').value);
  $('sim-clock').textContent = timeLabel(result?.simulation_end_time || f.time || 0);
  const duration = state.options?.duration || Number($('duration').value);
  const progress = state.status === 'completed' || result ? 100 : Math.min(99, (f.time || 0) / duration * 100);
  $('run-progress').style.width = `${progress}%`;
  $('progress-label').textContent = f.time > duration && !result ? '남은 환자 이송 중' : `${Math.round(progress)}%`;
  $('episode-label').textContent = state.episode ? `에피소드 ${state.episode} / ${state.total_episodes || 1}` : '실행 대기';
  const hospitals = f.hospitals || state.hospitals || network?.hospitals;
  if (hospitals) renderHospitals(hospitals);
  $('virtual-mix-actual').hidden = !state.hospitals;
  if (state.hospitals) {
    const counts = virtualTypeLabels.map(type => state.hospitals.filter(h=>h.virtual && h.type===type).length);
    $('virtual-mix-actual').textContent = `${active ? '현재' : '직전'} 실행 생성: ${virtualMixLabel(counts)}`;
  }
  if (f.time !== renderState.lastFrame || state.frame_revision !== renderState.lastFrameRevision || state.episode !== renderState.lastFrameEpisode || ['completed', 'stopped', 'error'].includes(state.status)) {
    const previous = new Map(vehicles.map(vehicle => [vehicle.id, vehicle]));
    vehicles = (f.vehicles || []).map(vehicle => {
      const old = previous.get(vehicle.id);
      const trail = trails.get(vehicle.id) || [];
      trail.push([vehicle.x, vehicle.y]); if(trail.length > 35) trail.shift(); trails.set(vehicle.id, trail);
      return {...vehicle, fromX: old?.x ?? vehicle.x, fromY: old?.y ?? vehicle.y, at: performance.now()};
    });
    const live = new Set(vehicles.map(v => v.id));
    for (const id of trails.keys()) if (!live.has(id)) trails.delete(id);
    mapPatients = f.patients || [];
    const previousTraffic = new Map(trafficVehicles.map(car => [car.id,car]));
    trafficVehicles = (f.traffic || []).map(car => {
      const old = previousTraffic.get(car.id);
      return {...car, fromX: old?.x ?? car.x, fromY: old?.y ?? car.y, at: performance.now()};
    });
    nearbyLanes = f.lanes || [];
    renderState.lastFrame = f.time; renderState.lastFrameEpisode = state.episode;
    renderState.lastFrameRevision = state.frame_revision;
  }
  updateFollowControls();
  renderReward(state);
  window.patientResults?.stateChanged(state);
}

async function loadNetwork() {
  try {
    const data = await api('/api/network');
    if (data.loading) { setTimeout(loadNetwork, 1200); return; }
    network = data;
    for (const [kind, shape] of data.roads) {
      const path = roadPaths[kind];
      shape.forEach(([x,y], i) => { if(i) path.lineTo(x,y); else path.moveTo(x,y); });
    }
    mapHospitals = data.hospitals; resetMap();
    $('map-loading').hidden = true;
    renderHospitals(data.hospitals);
  } catch (error) {
    $('map-loading').innerHTML = `<b>지도를 불러오지 못했습니다.</b><span>${escapeHTML(error.message)}</span>`;
  }
}
async function poll() {
  if (pollBusy || document.hidden) return;
  pollBusy = true;
  try {
    renderState(await api('/api/state'));
    $('connection').className = 'connection connected'; $('connection').innerHTML = '<i></i> 로컬 엔진 연결됨';
  } catch (error) {
    $('connection').className = 'connection'; $('connection').innerHTML = '<i></i> 연결 끊김';
    $('start').disabled = true;
    $('run-message').querySelector('span').textContent = '로컬 서버 연결을 확인하세요. 다시 연결되면 화면이 갱신됩니다.';
  } finally { pollBusy = false; }
}
async function init() {
  updateMode(); renderChart([], false); loadNetwork();
  try { await refreshModels(); } catch(error) { toast(error.message); }
  await poll(); setInterval(poll, 300);
}
init();
