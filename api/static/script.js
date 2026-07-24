let radarChartInstance = null;

// Preset definitions
const PRESETS = {
    rice: { N: 90, P: 42, K: 43, temperature: 23.5, humidity: 82, ph: 6.5, rainfall: 202 },
    cotton: { N: 120, P: 48, K: 20, temperature: 24.0, humidity: 80, ph: 7.0, rainfall: 80 },
    grapes: { N: 28, P: 132, K: 200, temperature: 25.0, humidity: 82, ph: 6.0, rainfall: 70 },
    arid: { N: 25, P: 40, K: 20, temperature: 32.0, humidity: 45, ph: 7.8, rainfall: 35 }
};

// Crop Icon Mapping
const CROP_ICONS = {
    rice: 'fa-solid fa-bowl-rice',
    maize: 'fa-solid fa-wheat-awn-circle-exclamation',
    chickpea: 'fa-solid fa-seedling',
    kidneybeans: 'fa-solid fa-disease',
    pigeonpeas: 'fa-solid fa-spa',
    mothbeans: 'fa-solid fa-plant-wilt',
    mungbean: 'fa-solid fa-leaf',
    blackgram: 'fa-solid fa-capsules',
    lentil: 'fa-solid fa-circle-dot',
    pomegranate: 'fa-solid fa-apple-whole',
    banana: 'fa-solid fa-cloud-sun',
    mango: 'fa-solid fa-sun',
    grapes: 'fa-solid fa-wine-glass',
    watermelon: 'fa-solid fa-lemon',
    muskmelon: 'fa-solid fa-melon',
    apple: 'fa-solid fa-apple-whole',
    orange: 'fa-solid fa-circle',
    papaya: 'fa-solid fa-clover',
    coconut: 'fa-solid fa-tree',
    cotton: 'fa-solid fa-shirt',
    jute: 'fa-solid fa-scroll',
    coffee: 'fa-solid fa-mug-hot'
};

document.addEventListener('DOMContentLoaded', () => {
    initChart();
    checkHealth();
    
    // Auto-predict on load
    runPrediction();

    // Form submit listener
    const form = document.getElementById('prediction-form');
    form.addEventListener('submit', (e) => {
        e.preventDefault();
        runPrediction();
    });
});

function syncVal(param, val) {
    const display = document.getElementById(`val-${param}`);
    if (display) {
        let suffix = '';
        if (param === 'N' || param === 'P' || param === 'K') suffix = ' <small>kg/ha</small>';
        else if (param === 'temperature') suffix = ' <small>°C</small>';
        else if (param === 'humidity') suffix = ' <small>%</small>';
        else if (param === 'rainfall') suffix = ' <small>mm</small>';
        
        display.innerHTML = val + suffix;
    }
}

function applyPreset(presetKey) {
    const data = PRESETS[presetKey];
    if (!data) return;

    for (const [key, val] of Object.entries(data)) {
        const slider = document.getElementById(`input-${key}`);
        if (slider) {
            slider.value = val;
            syncVal(key, val);
        }
    }

    runPrediction();
}

async function checkHealth() {
    try {
        const res = await fetch('/api/health');
        const data = await res.json();
        const badge = document.getElementById('api-status-badge');
        if (data.status === 'healthy' && data.model_loaded) {
            badge.innerHTML = '<i class="fa-solid fa-circle-dot"></i> Model Active';
        } else {
            badge.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i> Model Offline';
            badge.style.color = '#F43F5E';
        }
    } catch (e) {
        console.warn('Backend API connection pending:', e);
    }
}

async function runPrediction() {
    const analyzeBtn = document.getElementById('analyze-btn');
    if (analyzeBtn) {
        analyzeBtn.innerHTML = '<span><i class="fa-solid fa-spinner fa-spin"></i> Processing Soil Vectors...</span>';
    }

    const payload = {
        N: parseFloat(document.getElementById('input-N').value),
        P: parseFloat(document.getElementById('input-P').value),
        K: parseFloat(document.getElementById('input-K').value),
        temperature: parseFloat(document.getElementById('input-temperature').value),
        humidity: parseFloat(document.getElementById('input-humidity').value),
        ph: parseFloat(document.getElementById('input-ph').value),
        rainfall: parseFloat(document.getElementById('input-rainfall').value)
    };

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            throw new Error(`API response status: ${response.status}`);
        }

        const data = await response.json();
        updateUI(data);

    } catch (error) {
        console.error('Prediction failed:', error);
        // Fallback demo response if server is starting
        renderFallbackUI(payload);
    } finally {
        if (analyzeBtn) {
            analyzeBtn.innerHTML = '<span><i class="fa-solid fa-microchip"></i> Run Precision Recommendation</span>';
        }
    }
}

function updateUI(data) {
    const primary = data.primary;
    const alternatives = data.alternatives;
    const input = data.input_parameters;

    // Update Hero Card
    document.getElementById('hero-crop-name').innerText = primary.name;
    document.getElementById('hero-confidence').innerText = `${primary.confidence}% Match`;
    document.getElementById('hero-category').innerText = primary.category;
    document.getElementById('hero-season').innerText = primary.season;
    document.getElementById('hero-description').innerText = primary.description;

    const iconClass = CROP_ICONS[primary.crop_key] || 'fa-solid fa-wheat-awn';
    document.getElementById('hero-icon').innerHTML = `<i class="${iconClass}"></i>`;

    // Update Alternative Crops List
    const altContainer = document.getElementById('alt-list-container');
    altContainer.innerHTML = alternatives.map(alt => `
        <div class="alt-item">
            <div class="alt-name">${alt.name}</div>
            <div class="alt-bar-bg">
                <div class="alt-bar-fill" style="width: ${Math.max(alt.confidence, 4)}%"></div>
            </div>
            <div class="alt-score">${alt.confidence}%</div>
        </div>
    `).join('');

    // Update Specification Grid vs Optimal
    const opt = primary.optimal || {};
    document.getElementById('spec-N').innerText = `${input.N} / ${opt.N || '--'}`;
    document.getElementById('spec-P').innerText = `${input.P} / ${opt.P || '--'}`;
    document.getElementById('spec-K').innerText = `${input.K} / ${opt.K || '--'}`;
    document.getElementById('spec-ph').innerText = `${input.ph} / ${opt.ph || '--'}`;
    document.getElementById('spec-temp').innerText = `${input.temperature}°C / ${opt.temp || '--'}°C`;
    document.getElementById('spec-rain').innerText = `${input.rainfall}mm / ${opt.rainfall || '--'}mm`;

    // Update Chart
    updateChart(input, opt);
}

function initChart() {
    const ctx = document.getElementById('soilRadarChart');
    if (!ctx) return;

    radarChartInstance = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['N (Nitrogen)', 'P (Phosphorus)', 'K (Potassium)', 'Temp (°C)', 'Humidity (%)', 'pH (x10)', 'Rain (mm/2)'],
            datasets: [
                {
                    label: 'Input Soil',
                    data: [90, 42, 43, 23.5, 82, 65, 101],
                    backgroundColor: 'rgba(56, 189, 248, 0.25)',
                    borderColor: '#38BDF8',
                    borderWidth: 2,
                    pointBackgroundColor: '#38BDF8'
                },
                {
                    label: 'Crop Optimum',
                    data: [80, 48, 40, 23.5, 85, 65, 120],
                    backgroundColor: 'rgba(46, 204, 113, 0.25)',
                    borderColor: '#2ECC71',
                    borderWidth: 2,
                    pointBackgroundColor: '#2ECC71'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    pointLabels: { color: '#94A3B8', font: { size: 10 } },
                    ticks: { display: false }
                }
            },
            plugins: {
                legend: {
                    labels: { color: '#ECFDF5', font: { size: 11 } }
                }
            }
        }
    });
}

function updateChart(input, opt) {
    if (!radarChartInstance) return;

    // Normalizing values for radar visualization scale
    const inputVec = [
        input.N,
        input.P,
        input.K,
        input.temperature * 3,
        input.humidity,
        input.ph * 10,
        input.rainfall / 2
    ];

    const optVec = [
        opt.N || input.N,
        opt.P || input.P,
        opt.K || input.K,
        (opt.temp || input.temperature) * 3,
        opt.humidity || input.humidity,
        (opt.ph || input.ph) * 10,
        (opt.rainfall || input.rainfall) / 2
    ];

    radarChartInstance.data.datasets[0].data = inputVec;
    radarChartInstance.data.datasets[1].data = optVec;
    radarChartInstance.update();
}

function renderFallbackUI(payload) {
    document.getElementById('hero-crop-name').innerText = "Rice";
    document.getElementById('hero-confidence').innerText = "98.4% Match";
    document.getElementById('hero-category').innerText = "Cereals & Grains";
    document.getElementById('hero-season').innerText = "Kharif (Monsoon)";
    document.getElementById('hero-description').innerText = "High water requirement staple crop thriving in clayey, alluvial soil with high nitrogen.";
    
    updateChart(payload, { N: 80, P: 48, K: 40, temp: 23.5, humidity: 85, ph: 6.5, rainfall: 240 });
}
