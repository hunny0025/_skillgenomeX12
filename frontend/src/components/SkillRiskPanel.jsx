import React, { useState, useCallback } from 'react';
import { Brain, AlertTriangle, CheckCircle2, TrendingUp, Zap, Activity, ShieldCheck, ShieldAlert, Gauge } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import axios from 'axios';


const INDICATORS = [
    { key: 'literacy_rate', label: 'Literacy Rate', min: 40, max: 100, step: 0.1, unit: '%', default: 72 },
    { key: 'internet_penetration', label: 'Internet Penetration', min: 5, max: 100, step: 0.1, unit: '%', default: 45 },
    { key: 'workforce_participation', label: 'Workforce Participation', min: 20, max: 80, step: 0.1, unit: '%', default: 55 },
    { key: 'urban_population', label: 'Urban Population', min: 5, max: 100, step: 0.1, unit: '%', default: 35 },
    { key: 'per_capita_income', label: 'Per Capita Income', min: 30000, max: 500000, step: 1000, unit: '₹', default: 120000 },
    { key: 'skill_training_count', label: 'Skill Training Beneficiaries', min: 1000, max: 250000, step: 1000, unit: '', default: 50000 }
];

const RISK_CONFIG = {
    Low: { color: 'text-emerald-400', bg: 'bg-emerald-500/10', border: 'border-emerald-500/30', glow: 'shadow-emerald-500/20', hex: '#10b981', gradient: 'from-emerald-600 to-emerald-400' },
    Moderate: { color: 'text-amber-400', bg: 'bg-amber-500/10', border: 'border-amber-500/30', glow: 'shadow-amber-500/20', hex: '#f59e0b', gradient: 'from-amber-600 to-amber-400' },
    High: { color: 'text-red-400', bg: 'bg-red-500/10', border: 'border-red-500/30', glow: 'shadow-red-500/20', hex: '#ef4444', gradient: 'from-red-600 to-red-400' }
};

const CHART_COLORS = ['#6366f1', '#3b82f6', '#06b6d4', '#10b981', '#f59e0b', '#f97316'];

const formatVal = (key, val) => {
    if (key === 'per_capita_income') return `₹${Number(val).toLocaleString()}`;
    if (key === 'skill_training_count') return Number(val).toLocaleString();
    return `${val}%`;
};

/* ─────────────── Animated Circular Score Ring ─────────────── */
const ScoreRing = ({ score, level }) => {
    const cfg = RISK_CONFIG[level] || RISK_CONFIG.Moderate;
    const radius = 60;
    const circumference = 2 * Math.PI * radius;
    const progress = (score / 100) * circumference;

    return (
        <div className="relative flex items-center justify-center">
            <svg width="160" height="160" className="-rotate-90">
                {/* Track */}
                <circle cx="80" cy="80" r={radius} fill="none" stroke="#1f2937" strokeWidth="10" />
                {/* Progress */}
                <motion.circle
                    cx="80" cy="80" r={radius}
                    fill="none"
                    stroke={cfg.hex}
                    strokeWidth="10"
                    strokeLinecap="round"
                    strokeDasharray={circumference}
                    initial={{ strokeDashoffset: circumference }}
                    animate={{ strokeDashoffset: circumference - progress }}
                    transition={{ duration: 1.2, ease: "easeOut" }}
                />
            </svg>
            <div className="absolute text-center">
                <motion.div
                    className={`text-4xl font-black leading-none ${cfg.color}`}
                    initial={{ opacity: 0, scale: 0.5 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: 0.4, duration: 0.5, type: 'spring' }}
                >
                    {score.toFixed(1)}
                </motion.div>
                <div className="text-[10px] text-gray-500 uppercase tracking-widest mt-1">Risk Score</div>
            </div>
        </div>
    );
};

/* ─────────────── Horizontal Risk Gauge Bar ─────────────── */
const RiskGaugeBar = ({ score, level }) => {
    const cfg = RISK_CONFIG[level] || RISK_CONFIG.Moderate;
    return (
        <div className="w-full">
            <div className="flex justify-between items-center text-xs text-gray-500 mb-1.5">
                <span>0</span>
                <span className={`font-bold uppercase tracking-wider ${cfg.color}`}>{level} Risk Zone</span>
                <span>100</span>
            </div>
            <div className="h-3 bg-gray-800 rounded-full overflow-hidden relative">
                {/* Zone markers */}
                <div className="absolute inset-y-0 left-[30%] w-px bg-gray-600/40 z-10" />
                <div className="absolute inset-y-0 left-[65%] w-px bg-gray-600/40 z-10" />
                {/* Fill */}
                <motion.div
                    className={`h-full rounded-full bg-gradient-to-r ${cfg.gradient}`}
                    initial={{ width: 0 }}
                    animate={{ width: `${Math.min(100, score)}%` }}
                    transition={{ duration: 0.9, ease: 'easeOut' }}
                />
            </div>
            <div className="flex justify-between text-[10px] text-gray-600 mt-1">
                <span>Low</span>
                <span>Moderate</span>
                <span>High</span>
            </div>
        </div>
    );
};

/* ─────────────── Custom Tooltip ─────────────── */
const CustomTooltip = ({ active, payload }) => {
    if (active && payload?.length) {
        return (
            <div className="bg-gray-950 border border-gray-700 rounded-lg p-2.5 text-xs text-gray-200 shadow-xl">
                <p className="font-bold mb-0.5 text-white">{payload[0].payload.name}</p>
                <p>Contribution: <span className="text-blue-400 font-bold">{payload[0].value.toFixed(2)}%</span></p>
            </div>
        );
    }
    return null;
};

/* ─────────────── Loading Pulse Animation ─────────────── */
const LoadingPulse = () => (
    <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="flex flex-col items-center justify-center py-16"
    >
        <div className="relative">
            <motion.div
                className="w-20 h-20 rounded-full border-4 border-purple-500/30"
                animate={{ scale: [1, 1.3, 1], opacity: [0.3, 0.1, 0.3] }}
                transition={{ duration: 1.5, repeat: Infinity }}
            />
            <div className="absolute inset-0 flex items-center justify-center">
                <Brain size={28} className="text-purple-400 animate-pulse" />
            </div>
        </div>
        <p className="text-sm text-gray-400 mt-4 animate-pulse">AI model processing…</p>
    </motion.div>
);

/* ═════════════════════════════════════════════════════════ */
/*                     MAIN COMPONENT                       */
/* ═════════════════════════════════════════════════════════ */
const SkillRiskPanel = () => {
    const initVals = Object.fromEntries(INDICATORS.map(i => [i.key, i.default]));
    const [values, setValues] = useState(initVals);
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleSlider = useCallback((key, val) => {
        setValues(prev => ({ ...prev, [key]: Number(val) }));
    }, []);

    const handlePredict = async () => {
        setLoading(true);
        setError(null);
        setResult(null);
        try {
            const res = await axios.post('/api/predict-skill-risk', values);
            setResult(res.data);

        } catch (e) {
            setError('Prediction failed. Ensure the backend server is running on port 5000.');
        } finally {
            setLoading(false);
        }
    };

    const chartData = result?.feature_contributions
        ? Object.entries(result.feature_contributions)
            .map(([k, v], i) => ({
                name: k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
                value: Math.abs(v),
                color: CHART_COLORS[i % CHART_COLORS.length]
            }))
            .sort((a, b) => b.value - a.value)
        : [];

    const riskCfg = RISK_CONFIG[result?.risk_level] || null;

    return (
        <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-6"
        >
            {/* ── Header ── */}
            <div className="flex items-center justify-between">
                <div>
                    <h2 className="text-2xl font-bold text-white flex items-center gap-2">
                        <Brain size={24} className="text-purple-400" /> Skill Risk Predictor
                    </h2>
                    <p className="text-sm text-gray-400 mt-0.5">Real-data AI model predicting unemployment & skill risk for any socio-economic profile</p>
                </div>
                {result && (
                    <motion.div
                        initial={{ opacity: 0, x: 10 }}
                        animate={{ opacity: 1, x: 0 }}
                        className="flex items-center gap-1.5 text-xs text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-3 py-1.5 rounded-full"
                    >
                        <CheckCircle2 size={12} /> Inference Complete
                    </motion.div>
                )}
            </div>

            {/* ── Input Sliders ── */}
            <div className="bg-gray-900 border border-gray-800 rounded-xl p-5 space-y-5">
                <h3 className="text-sm font-bold text-gray-300 uppercase tracking-wider flex items-center gap-2">
                    <Activity size={15} className="text-blue-400" /> Socio-Economic Indicators
                </h3>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-5">
                    {INDICATORS.map(ind => (
                        <div key={ind.key}>
                            <div className="flex justify-between text-xs mb-1.5">
                                <label className="text-gray-300">{ind.label}</label>
                                <span className="font-bold text-blue-400 tabular-nums">{formatVal(ind.key, values[ind.key])}</span>
                            </div>
                            <input
                                type="range"
                                min={ind.min}
                                max={ind.max}
                                step={ind.step}
                                value={values[ind.key]}
                                onChange={e => handleSlider(ind.key, e.target.value)}
                                className="w-full h-1.5 rounded-full appearance-none cursor-pointer"
                                style={{
                                    background: `linear-gradient(to right, #6366f1 ${((values[ind.key] - ind.min) / (ind.max - ind.min)) * 100}%, #374151 0%)`
                                }}
                            />
                            <div className="flex justify-between text-[10px] text-gray-600 mt-0.5">
                                <span>{formatVal(ind.key, ind.min)}</span>
                                <span>{formatVal(ind.key, ind.max)}</span>
                            </div>
                        </div>
                    ))}
                </div>

                <button
                    onClick={handlePredict}
                    disabled={loading}
                    className="w-full py-3.5 rounded-xl bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed text-white font-bold text-sm flex items-center justify-center gap-2 transition-all shadow-lg shadow-purple-500/20 hover:shadow-purple-500/30"
                >
                    {loading
                        ? <><Zap size={16} className="animate-bounce" /> Processing with AI Engine…</>
                        : <><Brain size={16} /> Predict Skill Risk</>}
                </button>

                {error && (
                    <motion.div
                        initial={{ opacity: 0, y: -4 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="p-3 bg-red-500/10 border border-red-500/30 rounded-lg flex gap-2"
                    >
                        <AlertTriangle size={14} className="text-red-400 shrink-0 mt-0.5" />
                        <p className="text-xs text-red-300">{error}</p>
                    </motion.div>
                )}
            </div>

            {/* ═══════════════════════  RESULT PANEL  ═══════════════════════ */}
            <AnimatePresence mode="wait">
                {loading && (
                    <motion.div key="loading"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="bg-gray-900 border border-gray-800 rounded-xl"
                    >
                        <LoadingPulse />
                    </motion.div>
                )}

                {!loading && result && (
                    <motion.div
                        key="result-panel"
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -10 }}
                        transition={{ duration: 0.5, ease: 'easeOut' }}
                        className={`bg-gray-900 border rounded-2xl overflow-hidden ${riskCfg?.border ?? 'border-gray-800'} shadow-2xl ${riskCfg?.glow ?? ''}`}
                    >
                        {/* Result Header Bar */}
                        <div className={`px-6 py-3 ${riskCfg?.bg ?? 'bg-gray-800/50'} border-b ${riskCfg?.border ?? 'border-gray-800'} flex items-center justify-between`}>
                            <div className="flex items-center gap-2">
                                <Gauge size={16} className={riskCfg?.color ?? 'text-gray-400'} />
                                <span className="text-sm font-bold text-white uppercase tracking-wider">AI Prediction Result</span>
                            </div>
                            <div className="flex items-center gap-2">
                                {!result.is_anomaly ? (
                                    <span className="inline-flex items-center gap-1 text-xs font-bold text-emerald-400 bg-emerald-500/15 px-2.5 py-0.5 rounded-full border border-emerald-500/30">
                                        <ShieldCheck size={11} /> Normal Input
                                    </span>
                                ) : (
                                    <span className="inline-flex items-center gap-1 text-xs font-bold text-orange-400 bg-orange-500/15 px-2.5 py-0.5 rounded-full border border-orange-500/30 animate-pulse">
                                        <ShieldAlert size={11} /> Unusual Input Detected
                                    </span>
                                )}
                            </div>
                        </div>

                        {/* Main Result Body */}
                        <div className="p-6">
                            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-center">
                                {/* LEFT: Score Ring */}
                                <div className="flex flex-col items-center">
                                    <ScoreRing score={result.skill_risk_score} level={result.risk_level} />
                                    <motion.div
                                        initial={{ opacity: 0 }}
                                        animate={{ opacity: 1 }}
                                        transition={{ delay: 0.8 }}
                                        className={`mt-3 text-lg font-black uppercase tracking-wider ${riskCfg?.color ?? 'text-white'}`}
                                    >
                                        {result.risk_level} Risk
                                    </motion.div>
                                </div>

                                {/* CENTER: Key Metrics Grid */}
                                <div className="lg:col-span-2 space-y-4">
                                    <div className="grid grid-cols-3 gap-3">
                                        {/* Predicted Unemployment */}
                                        <motion.div
                                            initial={{ opacity: 0, y: 8 }}
                                            animate={{ opacity: 1, y: 0 }}
                                            transition={{ delay: 0.2 }}
                                            className="bg-gray-800/60 border border-gray-700/50 rounded-xl p-4 text-center"
                                        >
                                            <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">Predicted Unemployment</div>
                                            <div className="text-3xl font-black text-white">{result.predicted_unemployment}<span className="text-base text-gray-400 ml-0.5">%</span></div>
                                        </motion.div>

                                        {/* Skill Risk Score */}
                                        <motion.div
                                            initial={{ opacity: 0, y: 8 }}
                                            animate={{ opacity: 1, y: 0 }}
                                            transition={{ delay: 0.35 }}
                                            className={`${riskCfg?.bg ?? ''} border ${riskCfg?.border ?? 'border-gray-700'} rounded-xl p-4 text-center`}
                                        >
                                            <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">Risk Score</div>
                                            <div className={`text-3xl font-black ${riskCfg?.color ?? 'text-white'}`}>{result.skill_risk_score.toFixed(1)}<span className="text-base text-gray-400 ml-0.5">/ 100</span></div>
                                        </motion.div>

                                        {/* Risk Level Badge */}
                                        <motion.div
                                            initial={{ opacity: 0, y: 8 }}
                                            animate={{ opacity: 1, y: 0 }}
                                            transition={{ delay: 0.5 }}
                                            className="bg-gray-800/60 border border-gray-700/50 rounded-xl p-4 text-center flex flex-col items-center justify-center"
                                        >
                                            <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">Risk Level</div>
                                            <div className={`text-2xl font-black ${riskCfg?.color ?? 'text-white'}`}>{result.risk_level}</div>
                                        </motion.div>
                                    </div>

                                    {/* Horizontal Gauge Bar */}
                                    <motion.div
                                        initial={{ opacity: 0 }}
                                        animate={{ opacity: 1 }}
                                        transition={{ delay: 0.6 }}
                                    >
                                        <RiskGaugeBar score={result.skill_risk_score} level={result.risk_level} />
                                    </motion.div>
                                </div>
                            </div>

                            {/* Anomaly Alert (full-width) */}
                            <AnimatePresence>
                                {result.is_anomaly && (
                                    <motion.div
                                        initial={{ opacity: 0, height: 0, marginTop: 0 }}
                                        animate={{ opacity: 1, height: 'auto', marginTop: 16 }}
                                        exit={{ opacity: 0, height: 0 }}
                                        className="flex items-start gap-3 p-4 bg-orange-500/10 border border-orange-500/30 rounded-xl"
                                    >
                                        <AlertTriangle size={18} className="text-orange-400 shrink-0 mt-0.5" />
                                        <div>
                                            <p className="text-sm font-bold text-orange-300">Unusual Input Detected</p>
                                            <p className="text-xs text-orange-200/70 mt-0.5">
                                                This socio-economic profile is statistically unusual compared to the training data. The prediction may be less reliable. Please verify the input values for accuracy.
                                            </p>
                                        </div>
                                    </motion.div>
                                )}
                            </AnimatePresence>

                            {/* Normal Status Banner (only if NOT anomaly) */}
                            {!result.is_anomaly && (
                                <motion.div
                                    initial={{ opacity: 0 }}
                                    animate={{ opacity: 1 }}
                                    transition={{ delay: 0.9 }}
                                    className="mt-4 flex items-center justify-between text-xs"
                                >
                                    <div className="flex items-center gap-1.5 text-emerald-400">
                                        <ShieldCheck size={13} />
                                        <span>Input profile is statistically normal</span>
                                    </div>
                                    <span className="text-gray-600">{result.model_used}</span>
                                </motion.div>
                            )}
                        </div>
                    </motion.div>
                )}

                {!loading && !result && (
                    <motion.div
                        key="placeholder"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="bg-gray-900 border border-dashed border-gray-700 rounded-xl py-16 flex flex-col items-center justify-center text-gray-600"
                    >
                        <Brain size={40} className="mb-3 opacity-20" />
                        <p className="text-sm">Adjust the indicators above and click <strong className="text-gray-400">Predict Skill Risk</strong></p>
                        <p className="text-xs text-gray-700 mt-1">Results will appear here with risk analysis</p>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* ── Feature Contribution Chart ── */}
            <AnimatePresence>
                {chartData.length > 0 && (
                    <motion.div
                        initial={{ opacity: 0, y: 12 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.3 }}
                        className="bg-gray-900 border border-gray-800 rounded-xl p-6"
                    >
                        <h3 className="text-sm font-bold text-gray-300 uppercase tracking-wider mb-5 flex items-center gap-2">
                            <TrendingUp size={15} className="text-blue-400" /> Feature Contributions to Risk Prediction
                        </h3>

                        {/* Inline bar chart (custom) */}
                        <div className="space-y-3 mb-6">
                            {chartData.map((item, i) => (
                                <div key={item.name} className="flex items-center gap-3">
                                    <div className="w-40 text-xs text-gray-400 text-right shrink-0 truncate">{item.name}</div>
                                    <div className="flex-1 h-5 bg-gray-800 rounded-full overflow-hidden relative">
                                        <motion.div
                                            className="h-full rounded-full"
                                            style={{ backgroundColor: item.color }}
                                            initial={{ width: 0 }}
                                            animate={{ width: `${Math.min(100, (item.value / (chartData[0]?.value || 1)) * 100)}%` }}
                                            transition={{ duration: 0.7, delay: i * 0.08, ease: 'easeOut' }}
                                        />
                                    </div>
                                    <div className="w-14 text-xs font-bold text-gray-300 tabular-nums text-right">{item.value.toFixed(1)}%</div>
                                </div>
                            ))}
                        </div>

                        {/* Recharts visual */}
                        <div className="border-t border-gray-800 pt-4">
                            <ResponsiveContainer width="100%" height={180}>
                                <BarChart data={chartData} margin={{ top: 4, right: 4, left: 0, bottom: 24 }}>
                                    <XAxis
                                        dataKey="name"
                                        tick={{ fill: '#6b7280', fontSize: 10 }}
                                        angle={-20}
                                        textAnchor="end"
                                        interval={0}
                                    />
                                    <YAxis tick={{ fill: '#6b7280', fontSize: 10 }} />
                                    <Tooltip content={<CustomTooltip />} />
                                    <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                                        {chartData.map((entry, i) => (
                                            <Cell key={i} fill={entry.color} />
                                        ))}
                                    </Bar>
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                        <p className="text-xs text-gray-600 text-center mt-2">Higher contribution = stronger influence on the unemployment prediction</p>
                    </motion.div>
                )}
            </AnimatePresence>
        </motion.div>
    );
};

export default SkillRiskPanel;
