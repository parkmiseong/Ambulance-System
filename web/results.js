'use strict';

(() => {
  const statusNames = {arrived: '도착', failed: '이송 실패', censored: '관측 종료'};
  const rewardNames = {golden: '골든타임', travel: '이동 시간', overtime: '시간 초과', mismatch: '병원 적합도', occupancy: '병원별 점유율', overcrowding: '과밀', rejection: '수용 거절', route_failure: '이송 실패'};
  const reasonNames = {arrived: '병원 도착', waiting_timeout: '배정 대기시간 초과', mission_timeout: '이송 제한시간 초과',
    vehicle_disappeared: '차량 운행 종료', horizon_timeout: '관측 시간 경계에서 운행 종료', horizon_waiting: '관측 시간 경계에서 대기 종료',
    no_valid_hospital: '선택 가능한 병원 없음', no_route: '이용 가능한 경로 없음', spawn_failed: '차량 생성 실패'};
  Object.assign(reasonNames, {no_route_after_rejection:'거절 후 재이송 경로 없음',
    redirect_failed:'거절 후 경로 변경 실패', hospital_stop_missing:'병원 정차 확인 실패'});
  const numeric = (value, digits = 0) => value == null ? '—' : fmt(value, digits);
  const duration = value => value == null ? '—' : `${fmt(value, 1)}초`;
  const badge = ktas => `<span class="severity-badge k${Number(ktas)}">KTAS ${Number(ktas)}</span>`;
  let page = 1, data = null, requestId = 0, refreshTimer, searchTimer, lastCount = -1, notifiedRun = null;
  let hospitalOptionsKey = '', episodeOptionsKey = '', sourceReady = false;

  function query() {
    const params = new URLSearchParams({page, limit: 15});
    for (const [key, id] of Object.entries({run: 'result-run', episode: 'filter-episode', ktas: 'filter-ktas', hospital: 'filter-hospital', status: 'filter-status', search: 'filter-search'})) {
      if ($(id).value) params.set(key, $(id).value);
    }
    return params;
  }

  function replaceOptions(id, entries, placeholder) {
    const selected = $(id).value;
    $(id).replaceChildren(new Option(placeholder, ''), ...entries.map(([value, label]) => new Option(label, value)));
    if (entries.some(([value]) => String(value) === selected)) $(id).value = selected;
  }

  async function sources(autoSelect = false) {
    try {
      const response = await api('/api/runs');
      const previous = $('result-run').value;
      replaceOptions('result-run', response.runs.map(run => [run.id, `${run.id.replace(/^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})_/, '$1.$2.$3 $4:$5:$6 · ')}${run.completed_episodes ? ` · ${run.completed_episodes}회 완료` : ' · 일부 기록'}`]), '현재 실행');
      if (autoSelect && !currentState.run_id && currentState.status === 'idle' && response.runs.length) {
        $('result-run').value = (response.runs.find(run => run.completed_episodes) || response.runs[0]).id;
      } else if (previous && $('result-run').value !== previous) {
        $('result-run').value = '';
      }
      sourceReady = true;
      await refresh();
    } catch (error) {
      $('result-source-note').textContent = error.message;
    }
  }

  async function refresh() {
    clearTimeout(refreshTimer);
    const serial = ++requestId, params = query();
    $('patient-csv').href = `/api/patients.csv?${params}`;
    try {
      const result = await api(`/api/patients?${params}`);
      if (serial !== requestId) return;
      data = result; page = result.page;
      render(result);
    } catch (error) {
      if (serial !== requestId) return;
      $('patient-table-body').innerHTML = `<tr><td colspan="9" class="table-empty">${escapeHTML(error.message)}</td></tr>`;
      $('result-source-note').textContent = '결과를 불러오지 못했습니다. 목록 새로고침으로 다시 시도하세요.';
    }
  }

  function schedule() {
    ++requestId; // Invalidate previous filters immediately, including the debounce window.
    clearTimeout(refreshTimer);
    refreshTimer = setTimeout(refresh, 150);
  }

  function render(result) {
    const saved = Boolean($('result-run').value);
    $('patient-record-count').textContent = `${fmt(result.record_count)}명`;
    $('result-source-note').textContent = `${saved ? `저장 결과 ${result.run_id}` : '현재 실행'} · ${(result.strategies || []).join(' / ') || '정책 대기'} · 결과 확정 ${fmt(result.record_count)}명`;
    const hKey = JSON.stringify(result.hospitals), eKey = JSON.stringify(result.episodes);
    if (hKey !== hospitalOptionsKey) {
      replaceOptions('filter-hospital', result.hospitals.map(h => [h.id, h.name]), '전체 병원'); hospitalOptionsKey = hKey;
    }
    if (eKey !== episodeOptionsKey) {
      replaceOptions('filter-episode', result.episodes.map(ep => [String(ep), `에피소드 ${ep}`]), '전체'); episodeOptionsKey = eKey;
    }
    $('severity-cards').innerHTML = result.severities.map(s => `<button class="severity-card ${$('filter-ktas').value === String(s.ktas) ? 'selected' : ''}" type="button" data-ktas="${s.ktas}" aria-label="KTAS ${s.ktas} 환자 ${s.total}명 결과 보기"><span class="severity-card-top"><i class="ktas-dot k${s.ktas}">${s.ktas}</i> KTAS ${s.ktas}</span><strong>${fmt(s.total)}<small>명</small></strong><div class="severity-card-bottom">도착 <b>${s.arrived}</b> · <span class="fail">실패 ${s.failed}</span> · 관측 종료 ${s.censored}<br>환자당 보상 <b>${numeric(s.avg_reward, 2)}</b></div></button>`).join('');
    const totals = [0, 0, 0, 0, 0];
    $('hospital-matrix-body').innerHTML = result.matrix.length ? result.matrix.map(h => {
      h.counts.forEach((count, i) => { totals[i] += count; });
      return `<tr><td>${escapeHTML(shortName(h.name))}<small>${escapeHTML(h.type || h.id)}</small></td>${h.counts.map((count, i) => `<td>${count ? `<button type="button" class="matrix-cell" data-hospital="${escapeHTML(h.id)}" data-ktas="${i+1}" aria-label="${escapeHTML(h.name)}에 도착한 KTAS ${i+1} 환자 ${count}명">${fmt(count)}</button>` : '<span class="matrix-zero">—</span>'}</td>`).join('')}<td><b>${fmt(h.arrived)}</b></td></tr>`;
    }).join('') + `<tr class="matrix-total"><td>도착 합계</td>${totals.map(count => `<td>${fmt(count)}</td>`).join('')}<td>${fmt(totals.reduce((a,b) => a+b, 0))}</td></tr>` : '<tr><td colspan="7" class="table-empty">아직 도착한 환자의 병원 기록이 없습니다.</td></tr>';
    $('patient-table-body').innerHTML = result.rows.length ? result.rows.map(row => `<tr><td><b>${escapeHTML(row.patient_id)}</b><small>에피소드 ${row.episode}</small></td><td>${badge(row.ktas)}</td><td>${escapeHTML(row.hospital_name ? shortName(row.hospital_name) : '미배정')}<small>${escapeHTML(row.hospital_type || '배정된 병원 없음')}</small>${row.rejection_count ? `<small class="rejection-mark">수용 거절 ${fmt(row.rejection_count)}회</small>` : ''}</td><td><span class="outcome-status ${row.status}">${statusNames[row.status]}</span>${row.golden_success ? '<small class="golden-mark">골든타임 내 도착</small>' : ''}</td><td>${duration(row.wait_seconds)}</td><td>${duration(row.travel_seconds)}</td><td>${duration(row.total_seconds)}</td><td>${numeric(row.reward, 2)}</td><td><button type="button" class="patient-detail-button" data-record="${escapeHTML(row.record_id)}" aria-label="에피소드 ${row.episode} 환자 ${escapeHTML(row.patient_id)} 상세">상세</button></td></tr>`).join('') : '<tr><td colspan="9" class="table-empty">조건에 맞는 결과가 없습니다. 실행하거나 필터를 조정하세요.</td></tr>';
    $('patient-pagination-note').textContent = result.total ? `${fmt(result.total)}명 중 ${fmt((result.page-1)*result.limit+1)}–${fmt(Math.min(result.page*result.limit,result.total))}명` : '0명의 결과';
    $('patient-page').textContent = `${result.page} / ${result.pages}`;
    $('patient-prev').disabled = result.page <= 1;
    $('patient-next').disabled = result.page >= result.pages;
    if (!$('patient-dialog').open) return;
    // Keep the user reading the selected record while live results arrive.
  }

  function detail(row) {
    const parts = row.reward_components || {}, occupancy = row.occupancy_at_selection;
    $('patient-dialog-title').textContent = `${row.patient_id} · 에피소드 ${row.episode}`;
    $('patient-dialog-content').innerHTML = `<div class="patient-detail-body"><div class="detail-badges">${badge(row.ktas)}<span class="outcome-status ${row.status}">${statusNames[row.status]}</span><span class="detail-policy">${escapeHTML(row.strategy_name || '정책 기록 없음')}</span></div><div class="patient-route"><span>${row.arrived ? '도착 병원' : '배정 병원 · 도착 여부는 이송 결과 확인'}</span><b>${escapeHTML(row.hospital_name || '병원 미배정')}</b><small>${escapeHTML(row.hospital_type || '배정된 병원 없음')}${row.hospital_id ? ` · ${escapeHTML(row.hospital_id)}` : ''}<br>구급차 ${escapeHTML(row.vehicle_id || '기록 없음 (이전 실행)')}</small></div><dl class="detail-grid"><div><dt>보고 시 Pre-KTAS</dt><dd>${row.pre_ktas == null ? '—' : row.pre_ktas}</dd></div><div><dt>배정 시 KTAS</dt><dd>${row.ktas}</dd></div><div><dt>총 소요 시간</dt><dd>${duration(row.total_seconds)}</dd></div><div><dt>배정 대기 시간</dt><dd>${duration(row.wait_seconds)}</dd></div><div><dt>배정 후 이송 시간</dt><dd>${duration(row.travel_seconds)}</dd></div><div><dt>연구용 골든타임</dt><dd>${row.golden_success ? '내 도착' : (row.arrived ? '시간 초과' : '도착 미확정/실패')}</dd></div><div><dt>선택 시 병원 점유율</dt><dd>${occupancy == null ? '—' : `${fmt(occupancy*100, 1)}%`}</dd></div><div><dt>병원 적합도 (0~1)</dt><dd>${numeric(row.hospital_suitability, 3)}</dd></div><div><dt>수용 거절 횟수</dt><dd>${numeric(row.rejection_count)}회</dd></div><div><dt>종료 사유</dt><dd class="small-value">${escapeHTML(reasonNames[row.reason] || row.reason || '—')}</dd></div></dl><div class="detail-timeline"><div><span>환자 발생</span><b>${row.occurrence_time == null ? '—' : timeLabel(row.occurrence_time)}</b></div><div><span>구급차 배정</span><b>${row.dispatch_time == null ? '미배정' : timeLabel(row.dispatch_time)}</b></div><div><span>결과 확정</span><b>${row.finish_time == null ? '—' : timeLabel(row.finish_time)}</b></div></div>${hospitalVisitDetails(row)}${congestionDetails(row)}${rejectionDetails(row)}<h3 class="detail-section-title">환자별 보상 내역</h3><div class="detail-rewards">${Object.entries(rewardNames).map(([key,name]) => `<div><span>${name}</span><b class="${(parts[key] || 0) < 0 ? 'negative' : ''}">${(parts[key] || 0) > 0 ? '+' : ''}${numeric(parts[key], 3)}</b></div>`).join('')}</div><div class="detail-reward-total"><span>총 보상</span><strong>${numeric(row.reward, 3)}</strong></div><p class="detail-footnote">시각은 해당 에피소드의 시뮬레이션 시계입니다. 병원 적합도·골든타임·보상은 현재 연구 모델의 가정값이며 생존율을 의미하지 않습니다. 이전 저장 기록에 없는 항목은 추정하지 않고 비워 둡니다.</p></div>`;
    $('patient-dialog').showModal();
  }

  function congestionDetails(row) {
    const actual = row.hospital_congestion_reward, occupancy = row.occupancy_at_selection;
    if (actual == null || occupancy == null) return '';
    const balanced = row.congestion_reward_mode === 'balanced';
    const rule = balanced ? '50% 미만 가점 · 50% 초과 감점 · 70% 초과 과밀 추가 감점' : '기존 점유율 비례 감점 · 70% 초과 과밀 추가 감점';
    const outcome = row.censored ? '관측 종료로 혼잡도 보상은 부여하지 않았습니다.'
      : !row.arrived && row.reward_version === 4 ? '이송 실패로 혼잡도 보상 대신 실패 비용을 적용했습니다.' : '';
    return `<section class="congestion-detail"><div><span>실제 적용 병원 혼잡도 보상</span><strong class="${actual > 0 ? 'positive' : ''}">${actual > 0 ? '+' : ''}${numeric(actual, 3)}점</strong></div><p>배정 직전 해당 병원의 점유율 ${fmt(occupancy * 100, 1)}%<br>${rule}${outcome ? `<br>${outcome}` : ''}</p></section>`;
  }

  function hospitalVisitDetails(row) {
    const visits = row.hospital_visits || [];
    if (!visits.length) return '';
    return `<h3 class="detail-section-title">실제 병원 방문·재이송 이력</h3><ol class="hospital-visit-history">${visits.map(visit => {
      const status = visit.reason === 'accepted_after_wait' ? `거절 후 대기 · ${timeLabel(visit.accepted_time)} 수용`
        : visit.accepted ? '수용' : `수용 거절 · ${visit.reason === 'hospital_full' ? '병상 부족' : visit.reason === 'not_accepting' ? '수용 중단' : escapeHTML(visit.reason)}`;
      return `<li><b>${escapeHTML(visit.hospital_name || visit.hospital_id)}</b><span class="${visit.accepted ? 'accepted' : 'refused'}">${status}</span><small>출발 ${timeLabel(visit.leg_dispatch_time)} → 도착 ${timeLabel(visit.arrival_time)} · 도착 시 점유율 ${visit.occupancy_at_arrival == null ? '—' : `${fmt(visit.occupancy_at_arrival * 100, 1)}%`}<br>구급차 ${escapeHTML(visit.vehicle_id)}</small></li>`;
    }).join('')}</ol>`;
  }

  function rejectionDetails(row) {
    const history = row.rejection_history || [];
    if (!history.length) return '';
    const reasons = {hospital_full:'병상 부족', not_accepting:'수용 중단'};
    return `<h3 class="detail-section-title">병원 수용 거절 이력 · ${numeric(row.rejection_count)}회</h3><ol class="rejection-history">${history.map(event => `<li><div><b>${escapeHTML(event.hospital_name || event.hospital_id)}</b><small>${timeLabel(event.time)} · ${event.stage === 'after_arrival' ? '도착 후 거절' : '사전 문의 거절 (이전 방식)'} · ${escapeHTML(reasons[event.reason] || event.reason)}</small></div><strong>${numeric(event.penalty, 1)}점</strong></li>`).join('')}</ol>`;
  }

  function filtersChanged() { page = 1; schedule(); }
  for (const id of ['filter-episode', 'filter-ktas', 'filter-hospital', 'filter-status']) $(id).addEventListener('change', filtersChanged);
  $('filter-search').addEventListener('input', () => { clearTimeout(searchTimer); searchTimer = setTimeout(filtersChanged, 250); });
  $('patient-filters').addEventListener('submit', event => { event.preventDefault(); filtersChanged(); });
  $('patient-filters').addEventListener('reset', () => { setTimeout(filtersChanged, 0); });
  $('result-run').addEventListener('change', () => { $('patient-filters').reset(); page = 1; schedule(); });
  $('refresh-runs').addEventListener('click', () => sources(false));
  $('patient-prev').addEventListener('click', () => { if (page > 1) { page--; refresh(); } });
  $('patient-next').addEventListener('click', () => { if (data && page < data.pages) { page++; refresh(); } });
  $('severity-cards').addEventListener('click', event => {
    const card = event.target.closest('[data-ktas]');
    if (card) { $('filter-ktas').value = $('filter-ktas').value === card.dataset.ktas ? '' : card.dataset.ktas; filtersChanged(); }
  });
  $('hospital-matrix-body').addEventListener('click', event => {
    const cell = event.target.closest('[data-hospital]');
    if (!cell) return;
    $('filter-hospital').value = cell.dataset.hospital; $('filter-ktas').value = cell.dataset.ktas;
    $('filter-status').value = 'arrived'; filtersChanged();
    $('patient-filters').scrollIntoView({behavior: reducedMotion ? 'auto' : 'smooth', block: 'center'});
  });
  $('patient-table-body').addEventListener('click', event => {
    const button = event.target.closest('[data-record]');
    if (button) { const row = data?.rows.find(row => row.record_id === button.dataset.record); if (row) detail(row); }
  });
  $('close-patient-dialog').addEventListener('click', () => $('patient-dialog').close());
  $('patient-dialog').addEventListener('click', event => { if(event.target === $('patient-dialog')) { const box = event.target.getBoundingClientRect(); if(event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) event.target.close(); } });
  window.patientResults = {stateChanged(state) {
    if (state.run_id && state.run_id !== notifiedRun) {
      notifiedRun = state.run_id; $('result-run').value = ''; $('patient-filters').reset(); page = 1; lastCount = -1;
      sources(false);
    }
    if (sourceReady && !$('result-run').value && state.patient_count !== lastCount) {
      lastCount = state.patient_count; schedule();
    }
  }};
  sources(true);
})();
