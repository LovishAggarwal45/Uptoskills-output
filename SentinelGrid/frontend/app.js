const qs = s => document.querySelector(s);
const canvas = qs('#map'), ctx = canvas.getContext('2d');

let currentPeople = [];
let currentBounds = { minX: -4, maxX: 11, minY: -10, maxY: 25 };

function toCanvas(gx, gy) {
  const padL = 72, padR = 36, padT = 36, padB = 44;
  const gw = 720 - padL - padR;
  const gh = 440 - padT - padB;
  const { minX, maxX, minY, maxY } = currentBounds;
  const cx = padL + ((gx - minX) / (maxX - minX)) * gw;
  const cy = padT + gh - ((gy - minY) / (maxY - minY)) * gh;
  return [cx, cy];
}

function draw(people = [], hovered = null) {
  currentPeople = people;
  
  // Calculate dynamic or venue bounds
  let minX = -4, maxX = 11, minY = -10, maxY = 25;
  if (people && people.length > 0) {
    people.forEach(p => {
      if (p.x < minX) minX = Math.floor(p.x) - 1;
      if (p.x > maxX) maxX = Math.ceil(p.x) + 1;
      if (p.y < minY) minY = Math.floor(p.y) - 1;
      if (p.y > maxY) maxY = Math.ceil(p.y) + 1;
    });
  }
  currentBounds = { minX, maxX, minY, maxY };

  const padL = 72, padR = 36, padT = 36, padB = 44;
  const gw = 720 - padL - padR;
  const gh = 440 - padT - padB;

  // Background
  ctx.fillStyle = '#061016';
  ctx.fillRect(0, 0, 720, 440);

  // Monitored venue ground area
  const [tlX, tlY] = toCanvas(minX, maxY);
  const [brX, brY] = toCanvas(maxX, minY);
  ctx.fillStyle = '#0a1a24';
  ctx.fillRect(tlX, tlY, brX - tlX, brY - tlY);
  ctx.strokeStyle = '#1a3340';
  ctx.lineWidth = 1;
  ctx.strokeRect(tlX, tlY, brX - tlX, brY - tlY);

  // Vertical grid lines (X axis)
  ctx.lineWidth = 1;
  for (let x = Math.ceil(minX); x <= Math.floor(maxX); x += 2) {
    const [cx] = toCanvas(x, minY);
    ctx.beginPath();
    ctx.strokeStyle = x === 0 ? '#264b59' : '#12252e';
    ctx.moveTo(cx, padT);
    ctx.lineTo(cx, padT + gh);
    ctx.stroke();

    ctx.fillStyle = x === 0 ? '#9ebbc4' : '#6b838c';
    ctx.font = '10px DM Mono';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText(`${x}m`, cx, padT + gh + 6);
  }

  // Horizontal grid lines (Y axis)
  for (let y = Math.ceil(minY); y <= Math.floor(maxY); y += 5) {
    const [, cy] = toCanvas(minX, y);
    ctx.beginPath();
    ctx.strokeStyle = y === 0 ? '#264b59' : '#12252e';
    ctx.moveTo(padL, cy);
    ctx.lineTo(padL + gw, cy);
    ctx.stroke();

    ctx.fillStyle = y === 0 ? '#9ebbc4' : '#6b838c';
    ctx.font = '10px DM Mono';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText(`${y}m`, padL - 8, cy);
  }

  // Axis Titles
  ctx.fillStyle = '#8aa3ab';
  ctx.font = '11px DM Mono';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';
  ctx.fillText("GROUND PLANE (metres) · MULTI-VIEW FUSED OCCUPANCY", 18, 14);

  ctx.font = '10px DM Mono';
  ctx.fillStyle = '#5c7882';
  ctx.textAlign = 'center';
  ctx.fillText("X: Piazza Width (metres)", padL + gw / 2, 440 - 14);

  ctx.save();
  ctx.translate(16, padT + gh / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("Y: Piazza Length (metres)", 0, 0);
  ctx.restore();

  // Camera locations
  const camLocations = {
    C1: [7.1, -2.0], C2: [0.4, 21.5], C3: [7.2, 14.1],
    C4: [5.6, -3.9], C5: [-0.9, 8.7], C6: [-0.0, -7.3], C7: [8.1, 3.7]
  };
  for (const [cid, [c_gx, c_gy]] of Object.entries(camLocations)) {
    const [cx, cy] = toCanvas(c_gx, c_gy);
    ctx.fillStyle = '#102733';
    ctx.strokeStyle = '#2d5363';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(cx, cy, 11, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = '#6ef2c2';
    ctx.font = 'bold 9px DM Mono';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(cid, cx, cy);
  }

  // Fused people markers
  people.forEach((p, i) => {
    const [cx, cy] = toCanvas(p.x, p.y);
    const isMulti = p.views > 1;
    const isHovered = hovered && hovered.id === p.id;

    if (isMulti) {
      ctx.beginPath();
      ctx.arc(cx, cy, isHovered ? 13 : 10, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(229, 255, 114, 0.28)';
      ctx.fill();
    }

    ctx.beginPath();
    ctx.arc(cx, cy, isHovered ? 9 : 7.5, 0, Math.PI * 2);
    ctx.fillStyle = isMulti ? '#e5ff72' : '#41c5ff';
    ctx.fill();
    ctx.strokeStyle = isHovered ? '#ffffff' : '#061016';
    ctx.lineWidth = isHovered ? 2 : 1;
    ctx.stroke();

    ctx.fillStyle = '#061016';
    ctx.font = 'bold 8.5px DM Mono';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(p.id || (i + 1), cx, cy);
  });

  // Hover Tooltip
  if (hovered) {
    const [cx, cy] = toCanvas(hovered.x, hovered.y);
    const tx = Math.min(Math.max(cx + 14, 80), 500);
    const ty = Math.min(Math.max(cy - 20, 30), 380);
    const text1 = `Person #${hovered.id} · (${hovered.x.toFixed(2)}m, ${hovered.y.toFixed(2)}m)`;
    const text2 = `Conf: ${(hovered.confidence * 100).toFixed(1)}% · Views: ${hovered.views} (${hovered.cameras ? hovered.cameras.join(', ') : 'CCTV'})`;

    ctx.fillStyle = 'rgba(9, 21, 29, 0.94)';
    ctx.strokeStyle = '#29d39a';
    ctx.lineWidth = 1;
    ctx.fillRect(tx, ty, 205, 42);
    ctx.strokeRect(tx, ty, 205, 42);

    ctx.fillStyle = '#e5ff72';
    ctx.font = 'bold 10px DM Mono';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText(text1, tx + 8, ty + 7);

    ctx.fillStyle = '#8aa3ab';
    ctx.font = '10px DM Mono';
    ctx.fillText(text2, tx + 8, ty + 24);
  }
}

// Initial draw with empty grid
draw();

// Mouse hover interaction on canvas
canvas.onmousemove = e => {
  if (!currentPeople.length) return;
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const mx = (e.clientX - rect.left) * scaleX;
  const my = (e.clientY - rect.top) * scaleY;

  let closest = null, minDist = 16;
  currentPeople.forEach(p => {
    const [cx, cy] = toCanvas(p.x, p.y);
    const d = Math.hypot(mx - cx, my - cy);
    if (d < minDist) {
      minDist = d;
      closest = p;
    }
  });
  draw(currentPeople, closest);
};

canvas.onmouseleave = () => {
  if (currentPeople.length) draw(currentPeople, null);
};

async function configure() {
  const cap = Number(qs('#limit').value) || 50;
  const loc = qs('#location').value || 'Demo venue';
  await fetch('/api/settings', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      capacity_limit: cap,
      location: loc,
      cluster_distance_m: 0.8,
      detection_conf_thresh: 0.30
    })
  });
}

qs('#run').onclick = async () => {
  const files = [...qs('#frames').files];
  if (!files.length) return alert('Select one or more camera images.');
  await configure();

  const ids = files.map((f, i) => {
    const m = f.name.match(/(C\d+)/i);
    return m ? m[1].toUpperCase() : `C${i + 1}`;
  });

  const data = new FormData();
  files.forEach(f => data.append('frames', f));
  ids.forEach(id => data.append('camera_ids', id));

  qs('#state').textContent = 'ANALYSING';
  qs('#state').style.borderColor = '#41c5ff';
  qs('#state').style.color = '#41c5ff';
  qs('#state').style.background = '#0a1d27';

  try {
    const r = await fetch('/api/analyze', { method: 'POST', body: data });
    const out = await r.json();
    if (!r.ok) throw Error(out.detail || 'Analysis failed');

    const e = out.event;
    const stats = out.stats || {};

    // Unified Count
    qs('#count').textContent = e.unified_count || e.count;

    // Model Confidence (mathematical mean of fused detections)
    qs('#confidence').textContent = (e.average_confidence * 100).toFixed(1) + '%';

    // Occupancy Risk (unified count / capacity)
    const occPct = e.occupancy_percentage !== undefined ? e.occupancy_percentage : (e.risk * 100);
    qs('#risk').textContent = occPct.toFixed(1) + '%';

    // Condition & Alert Status
    const isSafe = e.condition === 'SAFE' || e.condition === 'NORMAL';
    qs('#condition').textContent = e.condition;
    qs('#condition').style.color = isSafe ? '#6ef2c2' : '#ff8c7a';

    qs('#state').textContent = isSafe ? 'SAFE' : 'ALERT';
    qs('#state').style.borderColor = isSafe ? '#29d39a' : '#ff8c7a';
    qs('#state').style.color = isSafe ? '#6ef2c2' : '#ff8c7a';
    qs('#state').style.background = isSafe ? '#0c2927' : '#2b1210';

    // Draw Bird's-Eye Ground Plane with all fused people
    currentPeople = out.ground_plane_people || [];
    draw(currentPeople);

    // Render Camera Feeds with visible bounding boxes and detection counts
    qs('#views').innerHTML = out.views.map(v => `
      <figure>
        <img src="data:image/jpeg;base64,${v.image}" alt="${v.camera_id}">
        <figcaption>${v.camera_id} · ${v.detections} detections</figcaption>
      </figure>
    `).join('');

  } catch (err) {
    alert(err.message);
    qs('#state').textContent = 'ERROR';
    qs('#state').style.borderColor = '#ff8c7a';
    qs('#state').style.color = '#ff8c7a';
    qs('#state').style.background = '#2b1210';
  }
};
