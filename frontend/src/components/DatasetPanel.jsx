import React, { useState, useEffect, useRef } from 'react';
import { UploadCloud, Play, Database, CheckCircle2, AlertTriangle, XCircle, RefreshCw, Info } from 'lucide-react';
import axios from 'axios';

import { motion, AnimatePresence } from 'framer-motion';

const StatusBadge = ({ trained, source }) => {
    if (trained) {
        return (
            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold bg-emerald-500/15 text-emerald-400 border border-emerald-500/30">
                <CheckCircle2 size={11} /> Trained · {source}
            </span>
        );
    }
    return (
        <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold bg-yellow-500/15 text-yellow-400 border border-yellow-500/30">
            <AlertTriangle size={11} /> Not Trained
        </span>
    );
};

const DatasetPanel = () => {
    const [modelStatus, setModelStatus] = useState(null);
    const [training, setTraining] = useState(false);
    const [trainResult, setTrainResult] = useState(null);
    const [uploadResult, setUploadResult] = useState(null);
    const [uploading, setUploading] = useState(false);
    const [error, setError] = useState(null);
    const fileRef = useRef(null);

    const fetchStatus = async () => {
        try {
            const res = await axios.get('/api/model-status');
            setModelStatus(res.data);

        } catch (e) {
            console.error('model-status error', e);
        }
    };

    useEffect(() => {
        fetchStatus();
    }, []);

    const handleTrain = async (useUploaded = false) => {
        setTraining(true);
        setError(null);
        setTrainResult(null);
        try {
            const res = await axios.post('/api/train-model', { use_uploaded: useUploaded });
            const data = res.data;

            if (data.status === 'success') {
                setTrainResult(data);
                await fetchStatus();
            } else {
                setError(data.message || 'Training failed.');
            }
        } catch (e) {
            setError('Network error during training.');
        } finally {
            setTraining(false);
        }
    };

    const handleFileChange = async (e) => {
        const file = e.target.files?.[0];
        if (!file) return;
        setUploading(true);
        setUploadResult(null);
        setError(null);
        const form = new FormData();
        form.append('file', file);
        try {
            const res = await axios.post('/api/upload-dataset', form);
            const data = res.data;

            setUploadResult(data);
        } catch (e) {
            setError('Upload failed. Please try again.');
        } finally {
            setUploading(false);
            e.target.value = '';
        }
    };

    const importanceSorted = modelStatus?.feature_importances
        ? Object.entries(modelStatus.feature_importances).sort((a, b) => b[1] - a[1])
        : [];

    const maxImportance = importanceSorted[0]?.[1] || 1;

    return (
        <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-6"
        >
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h2 className="text-2xl font-bold text-white">Dataset & Model Training</h2>
                    <p className="text-sm text-gray-400 mt-0.5">Real socio-economic data powered AI for India</p>
                </div>
                <button
                    onClick={fetchStatus}
                    className="p-2 rounded-lg bg-gray-800 hover:bg-gray-700 text-gray-400 hover:text-white transition-colors"
                    title="Refresh status"
                >
                    <RefreshCw size={16} />
                </button>
            </div>

            {/* Status Cards */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {[
                    {
                        label: 'Model Status',
                        value: <StatusBadge trained={modelStatus?.trained} source={modelStatus?.data_source} />,
                        icon: <CheckCircle2 size={18} className={modelStatus?.trained ? 'text-emerald-400' : 'text-yellow-400'} />
                    },
                    {
                        label: 'R² Accuracy',
                        value: <span className="text-2xl font-bold text-emerald-400">{modelStatus?.r2_display ?? '—'}</span>,
                        icon: <Database size={18} className="text-blue-400" />
                    },
                    {
                        label: 'Dataset Rows',
                        value: <span className="text-2xl font-bold text-white">{modelStatus?.dataset_rows?.toLocaleString() ?? '—'}</span>,
                        icon: <Database size={18} className="text-purple-400" />
                    },
                    {
                        label: 'Data Source',
                        value: <span className="text-lg font-semibold text-blue-300 capitalize">{modelStatus?.data_source ?? '—'}</span>,
                        icon: <Info size={18} className="text-sky-400" />
                    }
                ].map((card, i) => (
                    <div key={i} className="bg-gray-900 border border-gray-800 rounded-xl p-4">
                        <div className="flex items-center justify-between mb-2">
                            <span className="text-xs text-gray-400 uppercase tracking-wider">{card.label}</span>
                            {card.icon}
                        </div>
                        {card.value}
                    </div>
                ))}
            </div>

            {/* Upload + Train Row */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Upload Section */}
                <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
                    <h3 className="text-sm font-bold text-gray-300 uppercase tracking-wider mb-4 flex items-center gap-2">
                        <UploadCloud size={16} className="text-blue-400" /> Upload Dataset
                    </h3>

                    <div
                        onClick={() => fileRef.current?.click()}
                        className="border-2 border-dashed border-gray-700 hover:border-blue-500/60 rounded-xl p-8 flex flex-col items-center justify-center cursor-pointer transition-all group"
                    >
                        <UploadCloud size={32} className="text-gray-600 group-hover:text-blue-400 mb-2 transition-colors" />
                        <p className="text-sm text-gray-400 group-hover:text-gray-200 transition-colors">
                            {uploading ? 'Uploading…' : 'Click to upload a CSV file'}
                        </p>
                        <p className="text-xs text-gray-600 mt-1">Required: State, Literacy_Rate, Internet_Penetration, etc.</p>
                        <input
                            ref={fileRef}
                            type="file"
                            accept=".csv"
                            className="hidden"
                            onChange={handleFileChange}
                        />
                    </div>

                    <AnimatePresence>
                        {uploadResult && (
                            <motion.div
                                initial={{ opacity: 0, y: 8 }}
                                animate={{ opacity: 1, y: 0 }}
                                exit={{ opacity: 0 }}
                                className={`mt-4 p-3 rounded-lg text-xs ${uploadResult.ready_to_train
                                    ? 'bg-emerald-500/10 border border-emerald-500/30 text-emerald-300'
                                    : 'bg-yellow-500/10 border border-yellow-500/30 text-yellow-300'}`}
                            >
                                <p className="font-bold mb-1">{uploadResult.filename} · {uploadResult.row_count} rows</p>
                                <p>{uploadResult.message}</p>
                                {uploadResult.missing_required_columns?.length > 0 && (
                                    <p className="mt-1 text-red-400">Missing: {uploadResult.missing_required_columns.join(', ')}</p>
                                )}
                            </motion.div>
                        )}
                    </AnimatePresence>
                </div>

                {/* Train Section */}
                <div className="bg-gray-900 border border-gray-800 rounded-xl p-5 flex flex-col">
                    <h3 className="text-sm font-bold text-gray-300 uppercase tracking-wider mb-4 flex items-center gap-2">
                        <Play size={16} className="text-emerald-400" /> Train Model
                    </h3>

                    <div className="flex-1 space-y-3">
                        <button
                            onClick={() => handleTrain(false)}
                            disabled={training}
                            className="w-full py-3 px-4 rounded-xl bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 disabled:cursor-not-allowed text-white font-semibold text-sm flex items-center justify-center gap-2 transition-all"
                        >
                            {training ? <><RefreshCw size={16} className="animate-spin" /> Training…</> : <><Play size={16} /> Train on Seed Data (India 28 States)</>}
                        </button>

                        {uploadResult?.ready_to_train && (
                            <button
                                onClick={() => handleTrain(true)}
                                disabled={training}
                                className="w-full py-3 px-4 rounded-xl bg-blue-600 hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed text-white font-semibold text-sm flex items-center justify-center gap-2 transition-all"
                            >
                                {training ? <><RefreshCw size={16} className="animate-spin" /> Training…</> : <><Play size={16} /> Train on Uploaded Dataset</>}
                            </button>
                        )}
                    </div>

                    {error && (
                        <div className="mt-4 p-3 bg-red-500/10 border border-red-500/30 rounded-lg flex gap-2">
                            <XCircle size={14} className="text-red-400 shrink-0 mt-0.5" />
                            <p className="text-xs text-red-300">{error}</p>
                        </div>
                    )}

                    <AnimatePresence>
                        {trainResult && (
                            <motion.div
                                initial={{ opacity: 0, y: 8 }}
                                animate={{ opacity: 1, y: 0 }}
                                exit={{ opacity: 0 }}
                                className="mt-4 space-y-3"
                            >
                                {/* Best Model Banner */}
                                <div className="p-3 bg-emerald-500/10 border border-emerald-500/30 rounded-lg flex items-center gap-2">
                                    <CheckCircle2 size={14} className="text-emerald-400 shrink-0" />
                                    <span className="text-xs font-bold text-emerald-400">
                                        ★ Best Model: {trainResult.best_model || 'GradientBoosting'}
                                    </span>
                                    {trainResult.elapsed_seconds && (
                                        <span className="ml-auto text-[10px] text-gray-500">{trainResult.elapsed_seconds}s</span>
                                    )}
                                </div>

                                {/* Metrics Cards: R², MAE, RMSE */}
                                <div className="grid grid-cols-3 gap-2">
                                    {[
                                        {
                                            label: 'R² Score',
                                            value: trainResult.r2_score ?? trainResult.r2_display ?? '—',
                                            color: 'text-emerald-400',
                                            bg: 'bg-emerald-500/10 border-emerald-500/20'
                                        },
                                        {
                                            label: 'MAE',
                                            value: trainResult.mae ?? trainResult.all_metrics?.[trainResult.best_model?.toLowerCase?.()]?.mae ?? '—',
                                            color: 'text-amber-400',
                                            bg: 'bg-amber-500/10 border-amber-500/20'
                                        },
                                        {
                                            label: 'RMSE',
                                            value: trainResult.all_metrics?.[
                                                Object.keys(trainResult.all_metrics || {}).find(k =>
                                                    (trainResult.best_model || '').toLowerCase().includes(k.replace('_', ''))
                                                ) || Object.keys(trainResult.all_metrics || {})[0]
                                            ]?.rmse ?? '—',
                                            color: 'text-blue-400',
                                            bg: 'bg-blue-500/10 border-blue-500/20'
                                        }
                                    ].map((m, i) => (
                                        <div key={i} className={`rounded-lg p-3 border ${m.bg} text-center`}>
                                            <div className="text-[9px] text-gray-500 uppercase font-bold tracking-wider">{m.label}</div>
                                            <div className={`text-lg font-bold font-mono ${m.color} mt-1`}>
                                                {typeof m.value === 'number' ? m.value.toFixed(m.label === 'R² Score' ? 4 : 2) : m.value}
                                            </div>
                                        </div>
                                    ))}
                                </div>

                                {/* Model Comparison Table */}
                                {trainResult.all_models && (
                                    <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-3">
                                        <div className="text-[10px] text-gray-400 uppercase font-bold tracking-wider mb-2">Model Comparison</div>
                                        <div className="space-y-2">
                                            {Object.entries(trainResult.all_metrics || trainResult.all_models).map(([key, data]) => {
                                                const r2 = typeof data === 'object' ? data.r2_score : data;
                                                const mae = typeof data === 'object' ? data.mae : null;
                                                const isBest = trainResult.best_model && trainResult.best_model.toLowerCase().replace(/\s/g, '_').includes(key.replace('_', ''));
                                                const displayName = key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
                                                return (
                                                    <div key={key} className={`flex items-center gap-3 p-2 rounded-lg ${isBest ? 'bg-emerald-500/10 border border-emerald-500/20' : ''}`}>
                                                        {isBest && <span className="text-emerald-400 text-xs">★</span>}
                                                        <span className="text-xs text-gray-300 w-28 shrink-0">{displayName}</span>
                                                        <div className="flex-1 h-2 bg-gray-800 rounded-full overflow-hidden">
                                                            <motion.div
                                                                initial={{ width: 0 }}
                                                                animate={{ width: `${Math.max(0, r2 * 100)}%` }}
                                                                transition={{ duration: 0.8 }}
                                                                className={`h-full rounded-full ${isBest ? 'bg-emerald-500' : 'bg-gray-600'}`}
                                                            />
                                                        </div>
                                                        <span className={`text-xs font-mono font-bold w-14 text-right ${isBest ? 'text-emerald-400' : 'text-gray-400'}`}>
                                                            {typeof r2 === 'number' ? (r2 * 100).toFixed(1) + '%' : r2}
                                                        </span>
                                                        {mae != null && (
                                                            <span className="text-[10px] text-gray-500 w-16 text-right">MAE {mae}</span>
                                                        )}
                                                    </div>
                                                );
                                            })}
                                        </div>
                                    </div>
                                )}

                                {/* Data Info */}
                                {trainResult.data_info && (
                                    <div className="flex gap-4 text-[10px] text-gray-500">
                                        <span>Samples: {trainResult.data_info.samples}</span>
                                        <span>Train: {trainResult.data_info.train_size}</span>
                                        <span>Test: {trainResult.data_info.test_size}</span>
                                        <span>Features: {trainResult.data_info.features}</span>
                                    </div>
                                )}
                            </motion.div>
                        )}
                    </AnimatePresence>
                </div>
            </div>

            {/* Feature Importance */}
            {importanceSorted.length > 0 && (
                <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
                    <h3 className="text-sm font-bold text-gray-300 uppercase tracking-wider mb-4">Feature Importance (trained model)</h3>
                    <div className="space-y-3">
                        {importanceSorted.map(([feat, val]) => (
                            <div key={feat}>
                                <div className="flex justify-between text-xs mb-1">
                                    <span className="text-gray-300">{feat.replace(/_/g, ' ')}</span>
                                    <span className="text-blue-400 font-bold">{val.toFixed(1)}%</span>
                                </div>
                                <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                                    <motion.div
                                        initial={{ width: 0 }}
                                        animate={{ width: `${(val / maxImportance) * 100}%` }}
                                        transition={{ duration: 0.8, ease: 'anticipate' }}
                                        className="h-full bg-gradient-to-r from-blue-600 to-blue-400 rounded-full"
                                    />
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Data Schema Reference */}
            <div className="bg-gray-900/60 border border-gray-800/60 rounded-xl p-4">
                <p className="text-xs text-gray-500 font-bold uppercase tracking-wider mb-2 flex items-center gap-1.5">
                    <Info size={12} /> Expected CSV Schema
                </p>
                <div className="flex flex-wrap gap-2">
                    {['State', 'Literacy_Rate', 'Internet_Penetration', 'Workforce_Participation',
                        'Urban_Population_Percent', 'Per_Capita_Income', 'Skill_Training_Count', 'Unemployment_Rate (target)'].map(col => (
                            <span key={col} className={`text-xs px-2 py-0.5 rounded font-mono border ${col.includes('target') ? 'bg-orange-500/10 border-orange-500/40 text-orange-400' : 'bg-gray-800 border-gray-700 text-gray-400'}`}>
                                {col}
                            </span>
                        ))}
                </div>
                <p className="text-xs text-gray-600 mt-2">Data sources: data.gov.in · Census of India · PLFS · TRAI · NSDC</p>
            </div>
        </motion.div>
    );
};

export default DatasetPanel;
