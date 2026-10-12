(function () {
  'use strict';
  function nearestObservation(times, target) {
    if (!Number.isFinite(target)) return -1;
    let index = -1, distance = Infinity;
    times.forEach((time, i) => {
      if (Number.isFinite(time) && Math.abs(time - target) < distance) {
        index = i;
        distance = Math.abs(time - target);
      }
    });
    return index;
  }
  if (typeof module !== 'undefined') module.exports = { nearestObservation };
  if (typeof document === 'undefined') return;

  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const animate = (element, frames, options) => {
    if (!reduced.matches && element.animate) element.animate(frames, options);
  };
  const sections = document.querySelectorAll('.section-heading, .method-list article, .case-heading, .case-facts > div');
  if ('IntersectionObserver' in window) {
    const reveal = new IntersectionObserver(entries => entries.forEach(entry => {
      if (!entry.isIntersecting) return;
      animate(entry.target, [{ opacity: .4, transform: 'translateY(14px)' }, { opacity: 1, transform: 'none' }], { duration: 650, easing: 'cubic-bezier(.2,.7,.2,1)' });
      reveal.unobserve(entry.target);
    }), { threshold: .15 });
    sections.forEach(section => reveal.observe(section));
    const chart = document.getElementById('chart');
    if (chart) {
      const revealCurve = new IntersectionObserver(entries => {
        if (!entries[0].isIntersecting) return;
        const curve = chart.querySelector('polyline');
        if (!curve) return;
        animate(curve, [
          { clipPath: 'inset(0 100% 0 0)' },
          { clipPath: 'inset(0 0% 0 0)' },
        ], { duration: 1400, easing: 'linear' });
        revealCurve.disconnect();
      }, { threshold: .5 });
      revealCurve.observe(chart);
    }
  }
  const result = document.getElementById('result');
  if (result) new MutationObserver(() => {
    if (!result.hidden) animate(result, [{ opacity: .55, transform: 'translateY(6px)' }, { opacity: 1, transform: 'none' }], { duration: 240, easing: 'ease-out' });
  }).observe(result, { attributes: true, attributeFilter: ['hidden'] });

  const tablist = document.querySelector('.mode-tabs');
  if (tablist) {
    const marker = document.createElement('span');
    marker.className = 'tab-marker';
    marker.setAttribute('aria-hidden', 'true');
    tablist.append(marker);
    const position = () => {
      const selected = tablist.querySelector('[aria-selected=true]');
      marker.style.width = selected.offsetWidth + 'px';
      marker.style.transform = `translateX(${selected.offsetLeft}px)`;
    };
    new MutationObserver(position).observe(tablist, { subtree: true, attributes: true, attributeFilter: ['aria-selected'] });
    new ResizeObserver(position).observe(tablist);
    position();
  }

  // A low-resolution, seamless noise texture provides atmosphere without a 3D runtime.
  const hero = document.querySelector('.hero');
  if (hero) {
    const canvas = document.createElement('canvas');
    canvas.className = 'hero-atmosphere';
    canvas.setAttribute('aria-hidden', 'true');
    const ctx = canvas.getContext('2d');
    if (ctx) {
      hero.prepend(canvas);
      const texture = document.createElement('canvas');
      texture.width = 256; texture.height = 96;
      const tx = texture.getContext('2d');
      const pixels = tx.createImageData(256, 96);
      for (let y = 0; y < 96; y++) for (let x = 0; x < 256; x++) {
        const u = x / 256 * Math.PI * 2, v = y / 96;
        const field = Math.sin(u * 3 + Math.sin(v * 9)) * .35 + Math.sin(u * 7 - v * 12) * .18 + Math.cos(u * 2 + v * 6) * .47;
        const i = (y * 256 + x) * 4;
        pixels.data[i] = 239; pixels.data[i + 1] = 244; pixels.data[i + 2] = 244;
        pixels.data[i + 3] = Math.max(0, field + .15) * 105 * Math.sin(v * Math.PI) ** 2;
      }
      tx.putImageData(pixels, 0, 0);
      let visible = false, paused = false, frame = 0, last = 0, phase = 0;
      const draw = () => {
        const w = canvas.width, h = canvas.height;
        ctx.clearRect(0, 0, w, h);
        const offset = phase % w;
        ctx.drawImage(texture, offset - w, 0, w, h);
        ctx.drawImage(texture, offset, 0, w, h);
      };
      const tick = time => {
        if (time - last >= 50) {
          phase += Math.min(time - last, 100) * .008;
          last = time; draw();
        }
        frame = requestAnimationFrame(tick);
      };
      const sync = () => {
        cancelAnimationFrame(frame); frame = 0;
        canvas.hidden = reduced.matches;
        if (visible && !paused && !document.hidden && !reduced.matches) { last = performance.now(); frame = requestAnimationFrame(tick); }
      };
      new ResizeObserver(() => {
        canvas.width = Math.min(hero.clientWidth, 1200);
        canvas.height = Math.round(hero.clientHeight * .72);
        draw();
      }).observe(hero);
      if ('IntersectionObserver' in window) new IntersectionObserver(entries => { visible = entries[0].isIntersecting; sync(); }).observe(hero);
      document.addEventListener('visibilitychange', sync);
      reduced.addEventListener('change', sync);
      const toggle = document.createElement('button');
      toggle.className = 'motion-toggle';
      toggle.type = 'button';
      toggle.textContent = '暂停动态';
      toggle.setAttribute('aria-pressed', 'false');
      toggle.addEventListener('click', () => {
        paused = !paused;
        toggle.textContent = paused ? '播放动态' : '暂停动态';
        toggle.setAttribute('aria-pressed', String(paused));
        sync();
      });
      reduced.addEventListener('change', () => { toggle.hidden = reduced.matches; });
      toggle.hidden = reduced.matches;
      hero.append(toggle);
    }
  }

  window.installTrendCursor = function (host, rows, geometry) {
    const { left, right, top, bottom, width, height, x, y, selected, series, formatTime } = geometry;
    const svg = host.querySelector('svg');
    const description = host.getAttribute('aria-label');
    const times = rows.map(row => new Date(row.timestamp).getTime());
    const ns = 'http://www.w3.org/2000/svg';
    const cursor = document.createElementNS(ns, 'g');
    cursor.setAttribute('class', 'trend-cursor');
    cursor.style.display = 'none';
    svg.append(cursor);
    const readout = document.createElement('div');
    readout.className = 'trend-readout';
    readout.hidden = true;
    const inspectorId = host.id + '-inspector';
    let inspector = document.getElementById(inspectorId);
    if (!inspector) {
      inspector = document.createElement('div');
      inspector.id = inspectorId;
      host.insertAdjacentElement('afterend', inspector);
    }
    inspector.replaceChildren(readout);
    host.tabIndex = 0;
    let index = rows.length - 1, keyboardActive = false;
    const show = i => {
      if (i < 0) return;
      index = i;
      const row = rows[i], px = x(times[i]);
      cursor.replaceChildren();
      const line = document.createElementNS(ns, 'line');
      Object.entries({ x1: px, x2: px, y1: top, y2: height - bottom, stroke: '#56685e', 'stroke-dasharray': '3 3' }).forEach(([key, value]) => line.setAttribute(key, value));
      cursor.append(line);
      const time = document.createElement('strong');
      time.textContent = formatTime(row.timestamp) + ' · 北京时间';
      readout.replaceChildren(time);
      series.filter(([key]) => selected.has(key)).forEach(([key, color, label]) => {
        const item = document.createElement('span');
        item.style.color = color;
        item.textContent = `${label}  ${Number.isFinite(row[key]) ? row[key] : '缺失'}${key === 'aqi' ? '' : ' µg/m³'}`;
        readout.append(item);
        if (Number.isFinite(row[key])) {
          const dot = document.createElementNS(ns, 'circle');
          Object.entries({ cx: px, cy: y(row[key]), r: 4, fill: color, stroke: '#fff', 'stroke-width': 2 }).forEach(([name, value]) => dot.setAttribute(name, value));
          cursor.append(dot);
        }
      });
      cursor.style.display = '';
      readout.hidden = false;
      host.setAttribute('aria-label', description + '。' + readout.textContent);
    };
    const hide = () => { show(rows.length - 1); cursor.style.display = 'none'; host.setAttribute('aria-label', description); };
    const point = event => {
      keyboardActive = false;
      const rect = host.getBoundingClientRect();
      const fraction = Math.max(0, Math.min(1, ((event.clientX - rect.left) * width / rect.width - left) / (width - left - right)));
      show(nearestObservation(times, Math.min(...times) + fraction * (Math.max(...times) - Math.min(...times))));
    };
    host.onpointermove = point;
    host.onpointerdown = point;
    host.onpointerleave = event => {
      if (!keyboardActive && event.pointerType === 'mouse') hide();
    };
    host.onfocus = () => show(index);
    host.onblur = hide;
    host.onkeydown = event => {
      if (!['ArrowLeft', 'ArrowRight', 'Home', 'End', 'Escape'].includes(event.key)) return;
      event.preventDefault();
      if (event.key === 'Escape') return hide();
      keyboardActive = true;
      show(event.key === 'Home' ? 0 : event.key === 'End' ? rows.length - 1 : Math.max(0, Math.min(rows.length - 1, index + (event.key === 'ArrowRight' ? 1 : -1))));
    };
    hide();
  };
})();
