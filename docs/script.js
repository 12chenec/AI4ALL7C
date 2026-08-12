const metricData = {
  recall: {
    label: 'Recall',
    values: [0.000, 0.682, 0.091, 0.227],
    note: 'Recall asks: of the 88 real surges, how many did the model catch?'
  },
  precision: {
    label: 'Precision',
    values: [0.000, 0.107, 0.533, 0.426],
    note: 'Precision asks: when the model calls a surge, how often is that alert correct?'
  },
  f1: {
    label: 'F1',
    values: [0.000, 0.185, 0.155, 0.296],
    note: 'F1 balances precision and recall. XGBoost has the strongest overall F1 score.'
  },
  roc: {
    label: 'ROC-AUC',
    values: [null, 0.803, 0.841, 0.871],
    note: 'ROC-AUC measures ranking across thresholds, but can look optimistic with heavily imbalanced classes.'
  },
  pr: {
    label: 'PR-AUC',
    values: [null, 0.192, 0.250, 0.299],
    note: 'PR-AUC is especially useful here: the no-skill floor is the surge prevalence, only 0.039.'
  },
  accuracy: {
    label: 'Accuracy',
    values: [0.961, 0.768, 0.962, 0.958],
    note: 'Accuracy is the trap: the do-nothing baseline reaches 0.961 because 96.1% of test weeks are not surges.'
  }
};

const metricTabs = document.querySelectorAll('.metric-tab');
const chartRows = document.querySelectorAll('.chart-row');
const chartNote = document.getElementById('chart-note');

function setMetric(metricKey) {
  const metric = metricData[metricKey];
  if (!metric) return;

  metricTabs.forEach((tab) => {
    const selected = tab.dataset.metric === metricKey;
    tab.classList.toggle('active', selected);
    tab.setAttribute('aria-pressed', String(selected));
  });

  chartRows.forEach((row, index) => {
    const value = metric.values[index];
    const bar = row.querySelector('.chart-bar');
    const output = row.querySelector('.chart-value');

    if (value === null) {
      bar.style.width = '0%';
      output.textContent = 'n/a';
      row.classList.add('is-na');
    } else {
      bar.style.width = `${Math.max(0, Math.min(1, value)) * 100}%`;
      output.textContent = value.toFixed(3);
      row.classList.remove('is-na');
    }
  });

  chartNote.textContent = metric.note;
}

metricTabs.forEach((tab) => {
  tab.addEventListener('click', () => setMetric(tab.dataset.metric));
});

const navToggle = document.querySelector('.nav-toggle');
const siteNav = document.querySelector('.site-nav');

if (navToggle && siteNav) {
  navToggle.addEventListener('click', () => {
    const isOpen = siteNav.classList.toggle('open');
    navToggle.setAttribute('aria-expanded', String(isOpen));
  });

  siteNav.querySelectorAll('a').forEach((link) => {
    link.addEventListener('click', () => {
      siteNav.classList.remove('open');
      navToggle.setAttribute('aria-expanded', 'false');
    });
  });
}

const revealItems = document.querySelectorAll('.reveal');
if ('IntersectionObserver' in window) {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add('is-visible');
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.12, rootMargin: '0px 0px -35px 0px' });

  revealItems.forEach((item) => observer.observe(item));
} else {
  revealItems.forEach((item) => item.classList.add('is-visible'));
}

document.getElementById('year').textContent = new Date().getFullYear();
setMetric('recall');
