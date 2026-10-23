import React, { useState, useEffect, useRef, useMemo } from 'react';

const DEFAULT_API_BASE = import.meta.env.VITE_C2_API_URL || 'http://127.0.0.1:8002';
const DEFAULT_ADMIN_KEY = import.meta.env.VITE_C2_ADMIN_KEY || 'florasense-admin';

// --- Clean Vector Icons ---
const Icons = {
  Leaf: () => (
    <svg className="w-5 h-5 inline-block" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z"/>
      <path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"/>
    </svg>
  ),
  ShieldAlert: () => (
    <svg className="w-5 h-5 inline-block" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
      <line x1="12" y1="8" x2="12" y2="12"/>
      <line x1="12" y1="16" x2="12.01" y2="16"/>
    </svg>
  ),
  CheckCircle2: () => (
    <svg className="w-5 h-5 inline-block" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/>
    </svg>
  ),
  AlertTriangle: () => (
    <svg className="w-5 h-5 inline-block" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/>
      <line x1="12" y1="9" x2="12" y2="13"/>
      <line x1="12" y1="17" x2="12.01" y2="17"/>
    </svg>
  ),
  ClipboardList: () => (
    <svg className="w-5 h-5 inline-block" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect width="8" height="4" x="8" y="2" rx="1" ry="1"/>
      <path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/>
      <path d="M12 11h4"/><path d="M12 16h4"/><path d="M8 11h.01"/><path d="M8 16h.01"/>
    </svg>
  ),
  Database: () => (
    <svg className="w-5 h-5 inline-block" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <ellipse cx="12" cy="5" rx="9" ry="3"/>
      <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/>
      <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>
    </svg>
  ),
  UploadCloud: () => (
    <svg className="w-8 h-8 text-emerald-600" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242"/>
      <path d="M12 12v9"/><path d="m16 16-4-4-4 4"/>
    </svg>
  ),
  RefreshCw: () => (
    <svg className="w-4 h-4 inline-block animate-spin" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/>
      <path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/>
      <path d="M8 16H3v5"/>
    </svg>
  )
};

export default function DiseasePage() {
  const [activeTab, setActiveTab] = useState('diagnose');
  const [apiBase] = useState(DEFAULT_API_BASE);
  const [adminApiKey] = useState(DEFAULT_ADMIN_KEY);

  // System Health
  const [health, setHealth] = useState(null);
  const [isHealthLoading, setIsHealthLoading] = useState(false);

  // Diagnostics State
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [species, setSpecies] = useState('Aglaonema Maria');
  const [plantId, setPlantId] = useState('AGL-PLANT-001');
  const [topK] = useState(3);
  const [submitToReview, setSubmitToReview] = useState(true);
  const [isDiagnosing, setIsDiagnosing] = useState(false);
  const [diagnosisResult, setDiagnosisResult] = useState(null);
  const [diagError, setDiagError] = useState(null);
  const filePickerRef = useRef(null);

  // Agronomist Review Queue State
  const [reviewFilter, setReviewFilter] = useState('ALL');
  const [reviewTickets, setReviewTickets] = useState([]);
  const [reviewStats, setReviewStats] = useState({ pending: 0, resolved: 0, total: 0 });
  const [selectedTicket, setSelectedTicket] = useState(null);
  const [resolveLabel, setResolveLabel] = useState('bacterial');
  const [reviewerName, setReviewerName] = useState('Agronomist');
  const [resolveNotes, setResolveNotes] = useState('');
  const [addToProto, setAddToProto] = useState(true);
  const [isResolving, setIsResolving] = useState(false);

  // Dynamic Prototypes State
  const [prototypes, setPrototypes] = useState([]);
  const [protoLoading, setProtoLoading] = useState(false);
  const [newClassName, setNewClassName] = useState('');
  const [newDisplayName, setNewDisplayName] = useState('');
  const [enrollFiles, setEnrollFiles] = useState([]);
  const [enrollResult, setEnrollResult] = useState(null);
  const [isEnrolling, setIsEnrolling] = useState(false);
  const [supportClassName, setSupportClassName] = useState('');
  const [supportFiles, setSupportFiles] = useState([]);
  const [isAddingSupport, setIsAddingSupport] = useState(false);

  // Health fetch
  const refreshHealth = async () => {
    setIsHealthLoading(true);
    try {
      const res = await fetch(`${apiBase}/health`, { signal: AbortSignal.timeout(3000) });
      if (res.ok) {
        const data = await res.json();
        setHealth(data);
      } else {
        setHealth(null);
      }
    } catch {
      setHealth(null);
    } finally {
      setIsHealthLoading(false);
    }
  };

  const fetchPrototypes = async () => {
    setProtoLoading(true);
    try {
      const res = await fetch(`${apiBase}/prototypes`);
      if (res.ok) {
        const data = await res.json();
        setPrototypes(data.prototypes || []);
      }
    } catch (err) {
      console.error("Failed to fetch prototypes", err);
    } finally {
      setProtoLoading(false);
    }
  };

  const fetchReviewQueue = async () => {
    try {
      const res = await fetch(`${apiBase}/review`);
      if (res.ok) {
        const data = await res.json();
        setReviewTickets(data.tickets || []);
        setReviewStats(data.stats || { pending: 0, resolved: 0, total: 0 });
      }
    } catch (err) {
      console.error("Failed to fetch review queue", err);
    }
  };

  useEffect(() => {
    refreshHealth();
    fetchPrototypes();
    fetchReviewQueue();
  }, [apiBase]);

  const handleImageSelect = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      setImageFile(file);
      setImagePreview(URL.createObjectURL(file));
      setDiagnosisResult(null);
      setDiagError(null);
    }
  };

  const handleRunDiagnosis = async () => {
    if (!imageFile) return;
    setIsDiagnosing(true);
    setDiagError(null);

    try {
      const formData = new FormData();
      formData.append('file', imageFile);
      if (species) formData.append('species', species);
      if (plantId) formData.append('plant_id', plantId);
      formData.append('top_k', topK.toString());
      formData.append('submit_to_review', submitToReview ? 'true' : 'false');

      const res = await fetch(`${apiBase}/diagnose`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.detail || `Server responded with HTTP ${res.status}`);
      }

      const data = await res.json();
      setDiagnosisResult(data);
      if (data.review_ticket_id) {
        fetchReviewQueue();
      }
    } catch (err) {
      setDiagError(err.message || 'Diagnosis failed. Check backend connection.');
    } finally {
      setIsDiagnosing(false);
      refreshHealth();
    }
  };

  const handleResolveTicket = async () => {
    if (!selectedTicket) return;
    setIsResolving(true);
    try {
      const payload = {
        label: resolveLabel,
        reviewer: reviewerName,
        notes: resolveNotes,
        add_to_prototype: addToProto
      };

      const res = await fetch(`${apiBase}/review/${selectedTicket.ticket_id}/resolve`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': adminApiKey
        },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Resolve failed: HTTP ${res.status}`);
      }
      await fetchReviewQueue();
      setSelectedTicket(null);
    } catch (err) {
      alert(err.message || 'Failed to resolve ticket.');
    } finally {
      setIsResolving(false);
      fetchPrototypes();
    }
  };

  const handleRegisterDisease = async (e) => {
    e.preventDefault();
    if (!newClassName.trim() || enrollFiles.length < 1) {
      alert("Please provide a valid class name and at least one reference photo.");
      return;
    }
    setIsEnrolling(true);
    setEnrollResult(null);

    const formData = new FormData();
    formData.append('class_name', newClassName);
    formData.append('display_name', newDisplayName || newClassName);
    Array.from(enrollFiles).forEach(f => formData.append('files', f));

    try {
      const res = await fetch(`${apiBase}/register-disease`, {
        method: 'POST',
        headers: { 'X-API-Key': adminApiKey },
        body: formData
      });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      setEnrollResult({ success: true, data });
      fetchPrototypes();
      setNewClassName('');
      setNewDisplayName('');
      setEnrollFiles([]);
    } catch (err) {
      alert(err.message || "Failed to enroll new disease prototype.");
    } finally {
      setIsEnrolling(false);
    }
  };

  const renderQualityBadge = (flag) => {
    switch (flag) {
      case 'PASS':
        return (
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold tracking-wider uppercase bg-emerald-100 text-emerald-800 border border-emerald-300">
            <Icons.CheckCircle2 /> PASS (Approved for Export)
          </span>
        );
      case 'FAIL':
        return (
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold tracking-wider uppercase bg-amber-100 text-amber-800 border border-amber-300">
            <Icons.AlertTriangle /> FAIL (Abiotic Defect - Reject)
          </span>
        );
      case 'QUARANTINE':
      default:
        return (
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold tracking-wider uppercase bg-rose-100 text-rose-800 border border-rose-300">
            <Icons.ShieldAlert /> QUARANTINE (Hold & Isolate)
          </span>
        );
    }
  };

  const filteredTickets = useMemo(() => {
    if (reviewFilter === 'ALL') return reviewTickets;
    return reviewTickets.filter(t => t.status === reviewFilter);
  }, [reviewTickets, reviewFilter]);

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 font-sans antialiased pb-16">
      {/* Top Standard Navigation Header */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-30 shadow-xs">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-emerald-700 flex items-center justify-center text-white shadow-md">
                <Icons.Leaf />
              </div>
              <div>
                <h1 className="text-lg font-bold text-slate-900 tracking-tight">FloraSense</h1>
                <p className="text-xs text-slate-500">Foliar Pathology & Export Quality Gate</p>
              </div>
            </div>

            {/* Standard Status Indicator */}
            <div className="flex items-center gap-3 text-xs">
              <div className="flex items-center gap-2 bg-slate-100 px-3 py-1.5 rounded-lg border border-slate-200">
                <span className={`w-2 h-2 rounded-full ${health?.status === 'ok' ? 'bg-emerald-500' : 'bg-rose-500'}`} />
                <span className="font-medium text-slate-600">
                  {health?.status === 'ok' ? 'System Online' : 'System Offline'}
                </span>
                {health?.prototypes_loaded && (
                  <span className="text-slate-400 font-mono">({health.prototypes_loaded} prototypes)</span>
                )}
              </div>
              <button
                onClick={refreshHealth}
                title="Refresh backend status"
                className="p-1.5 text-slate-400 hover:text-slate-600 rounded-md hover:bg-slate-100 transition"
              >
                {isHealthLoading ? <Icons.RefreshCw /> : '↻'}
              </button>
            </div>
          </div>

          {/* Clean Navigation Tabs */}
          <nav className="flex space-x-8 -mb-px">
            <button
              onClick={() => setActiveTab('diagnose')}
              className={`py-3 px-1 border-b-2 font-medium text-sm flex items-center gap-2 transition ${
                activeTab === 'diagnose'
                  ? 'border-emerald-600 text-emerald-700 font-semibold'
                  : 'border-transparent text-slate-500 hover:text-slate-700 hover:border-slate-300'
              }`}
            >
              <Icons.ShieldAlert /> Foliar Diagnosis
            </button>
            <button
              onClick={() => { setActiveTab('review'); fetchReviewQueue(); }}
              className={`py-3 px-1 border-b-2 font-medium text-sm flex items-center gap-2 transition relative ${
                activeTab === 'review'
                  ? 'border-emerald-600 text-emerald-700 font-semibold'
                  : 'border-transparent text-slate-500 hover:text-slate-700 hover:border-slate-300'
              }`}
            >
              <Icons.ClipboardList /> Agronomist Review Queue
              {reviewStats.pending > 0 && (
                <span className="ml-1 px-1.5 py-0.5 text-xs font-bold rounded-full bg-rose-500 text-white">
                  {reviewStats.pending}
                </span>
              )}
            </button>
            <button
              onClick={() => { setActiveTab('prototypes'); fetchPrototypes(); }}
              className={`py-3 px-1 border-b-2 font-medium text-sm flex items-center gap-2 transition ${
                activeTab === 'prototypes'
                  ? 'border-emerald-600 text-emerald-700 font-semibold'
                  : 'border-transparent text-slate-500 hover:text-slate-700 hover:border-slate-300'
              }`}
            >
              <Icons.Database /> Disease Prototypes
            </button>
          </nav>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-6">
        {/* ========================================================================= */}
        {/* TAB 1: DIAGNOSIS & EXPORT QUALITY GATE                                   */}
        {/* ========================================================================= */}
        {activeTab === 'diagnose' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
            {/* Left Section: Specimen Upload & Acquisition */}
            <div className="lg:col-span-5 space-y-5">
              <div className="bg-white rounded-xl p-5 border border-slate-200 shadow-xs">
                <h2 className="text-base font-bold text-slate-900 mb-3 flex items-center justify-between">
                  <span>Specimen Intake</span>
                </h2>

                {/* Dropzone Container */}
                <div
                  onClick={() => filePickerRef.current?.click()}
                  className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition ${
                    imagePreview ? 'border-emerald-400 bg-emerald-50/20' : 'border-slate-300 hover:border-emerald-500 bg-slate-50'
                  }`}
                >
                  <input
                    type="file"
                    ref={filePickerRef}
                    onChange={handleImageSelect}
                    accept="image/*"
                    className="hidden"
                  />
                  {imagePreview ? (
                    <div className="relative group">
                      <img
                        src={imagePreview}
                        alt="Foliar specimen"
                        className="max-h-64 mx-auto rounded-lg shadow-xs object-contain border border-slate-200"
                      />
                      <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 flex items-center justify-center text-white text-xs font-semibold rounded-lg transition">
                        Click to choose a different photo
                      </div>
                    </div>
                  ) : (
                    <div className="flex flex-col items-center justify-center py-6">
                      <Icons.UploadCloud />
                      <p className="mt-2 text-sm font-semibold text-slate-700">Click to upload or drag & drop</p>
                      <p className="text-xs text-slate-500 mt-1">High-resolution foliage image (JPG, PNG)</p>
                    </div>
                  )}
                </div>

                {/* Acquisition Metadata Inputs */}
                <div className="mt-4 space-y-3">
                  <div>
                    <label className="block text-xs font-medium text-slate-700 mb-1">Target Species</label>
                    <input
                      type="text"
                      value={species}
                      onChange={(e) => setSpecies(e.target.value)}
                      placeholder="e.g. Aglaonema Maria"
                      className="w-full px-3 py-2 text-sm border border-slate-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:outline-hidden"
                    />
                  </div>
                  <div>
                    <label className="block text-xs font-medium text-slate-700 mb-1">Plant / Batch Identifier</label>
                    <input
                      type="text"
                      value={plantId}
                      onChange={(e) => setPlantId(e.target.value)}
                      placeholder="e.g. AGL-BATCH-001"
                      className="w-full px-3 py-2 text-sm border border-slate-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:outline-hidden"
                    />
                  </div>

                  <div className="pt-2 border-t border-slate-100 flex items-center justify-between">
                    <div>
                      <span className="text-xs font-semibold text-slate-800">Review Queue Submission</span>
                      <p className="text-[11px] text-slate-500 leading-tight">Forward low-confidence samples to agronomists</p>
                    </div>
                    <label className="relative inline-flex items-center cursor-pointer">
                      <input
                        type="checkbox"
                        checked={submitToReview}
                        onChange={(e) => setSubmitToReview(e.target.checked)}
                        className="sr-only peer"
                      />
                      <div className="w-9 h-5 bg-slate-200 peer-focus:outline-hidden rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-slate-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-emerald-600"></div>
                    </label>
                  </div>
                </div>

                {/* Diagnose Trigger Button */}
                <div className="mt-5 flex gap-2">
                  <button
                    onClick={handleRunDiagnosis}
                    disabled={!imageFile || isDiagnosing}
                    className={`flex-1 py-2.5 px-4 rounded-lg font-semibold text-sm shadow-xs transition flex items-center justify-center gap-2 ${
                      !imageFile || isDiagnosing
                        ? 'bg-slate-200 text-slate-400 cursor-not-allowed'
                        : 'bg-emerald-700 hover:bg-emerald-800 text-white'
                    }`}
                  >
                    {isDiagnosing ? (
                      <>
                        <Icons.RefreshCw /> Analyzing Specimen...
                      </>
                    ) : (
                      <>
                        <Icons.ShieldAlert /> Run Diagnosis
                      </>
                    )}
                  </button>
                  {imageFile && (
                    <button
                      onClick={() => {
                        setImageFile(null);
                        setImagePreview(null);
                        setDiagnosisResult(null);
                      }}
                      className="px-3 py-2.5 border border-slate-300 hover:bg-slate-100 text-slate-700 text-sm font-medium rounded-lg"
                    >
                      Clear
                    </button>
                  )}
                </div>

                {diagError && (
                  <div className="mt-3 p-3 text-xs bg-rose-50 border border-rose-200 text-rose-700 rounded-lg">
                    <strong>Error:</strong> {diagError}
                  </div>
                )}
              </div>
            </div>

            {/* Right Section: Comprehensive Diagnostic Verdict & Advisory */}
            <div className="lg:col-span-7">
              {diagnosisResult ? (
                <div className="space-y-5">
                  {/* Verdict Cards Header */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {/* Gate Verdict Card */}
                    <div className={`p-4 rounded-xl border ${
                      diagnosisResult.gate.status === 'CONFIRMED'
                        ? 'bg-emerald-50/50 border-emerald-300'
                        : 'bg-amber-50/60 border-amber-300'
                    }`}>
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-xs font-bold text-slate-500 uppercase tracking-wider">Gate Decision</span>
                        <span className="text-[11px] font-mono text-slate-400">tau = {diagnosisResult.gate.tau}</span>
                      </div>
                      <div className="flex items-baseline gap-2">
                        <span className={`text-xl font-extrabold ${
                          diagnosisResult.gate.status === 'CONFIRMED' ? 'text-emerald-800' : 'text-amber-800'
                        }`}>
                          {diagnosisResult.gate.status}
                        </span>
                        <span className="text-xs font-semibold text-slate-600">
                          ({(diagnosisResult.gate.confidence * 100).toFixed(1)}% confidence)
                        </span>
                      </div>
                      <p className="text-xs text-slate-600 mt-2">
                        {diagnosisResult.gate.accepted
                          ? "Automated diagnosis cleared the confidence threshold."
                          : "Confidence below safety gate. Routed to agronomist review queue."}
                      </p>
                    </div>

                    {/* Export Quality Card */}
                    <div className="p-4 rounded-xl border border-slate-200 bg-white flex flex-col justify-between">
                      <div>
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-xs font-bold text-slate-500 uppercase tracking-wider">Export Staging Action</span>
                        </div>
                        <div className="mt-1">
                          {renderQualityBadge(diagnosisResult.export_quality_flag)}
                        </div>
                      </div>
                      <div className="text-xs text-slate-500 mt-3 pt-2 border-t border-slate-100 flex justify-between">
                        <span>Predicted: <strong className="capitalize text-slate-700">{diagnosisResult.diagnosis.top_prediction}</strong></span>
                        {diagnosisResult.review_ticket_id && (
                          <span className="text-rose-600 font-mono font-medium">Ticket: #{diagnosisResult.review_ticket_id}</span>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Agronomic Advisory Protocol Card */}
                  <div className="bg-white rounded-xl p-5 border border-slate-200 shadow-xs">
                    <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500 mb-2 flex items-center gap-1.5">
                      <Icons.ShieldAlert /> Agronomic Advisory
                    </h3>
                    <p className="text-sm text-slate-800 leading-relaxed font-normal bg-slate-50 p-3.5 rounded-lg border border-slate-200">
                      {diagnosisResult.advisory}
                    </p>
                  </div>

                  {/* Probability Distribution */}
                  <div className="bg-white rounded-xl p-5 border border-slate-200 shadow-xs">
                    <div className="flex items-center justify-between mb-4">
                      <h3 className="text-sm font-bold text-slate-900">
                        Class Probabilities
                      </h3>
                      <span className="text-xs font-mono text-slate-400">T = {diagnosisResult.model.temperature}</span>
                    </div>

                    <div className="space-y-3">
                      {Object.entries(diagnosisResult.diagnosis.probabilities).map(([cName, prob]) => {
                        const pct = Math.max(0, Math.min(100, prob * 100));
                        const isWinning = cName === diagnosisResult.diagnosis.top_prediction;
                        return (
                          <div key={cName}>
                            <div className="flex justify-between items-center text-xs mb-1">
                              <span className={`font-semibold capitalize flex items-center gap-1.5 ${isWinning ? 'text-emerald-900' : 'text-slate-600'}`}>
                                {cName}
                                {isWinning && <span className="text-[10px] bg-emerald-100 text-emerald-800 px-1.5 py-0.2 rounded font-bold">TOP</span>}
                              </span>
                              <span className="font-mono text-slate-700 font-medium">{pct.toFixed(2)}%</span>
                            </div>
                            <div className="w-full bg-slate-100 rounded-full h-2.5 overflow-hidden">
                              <div
                                className={`h-2.5 rounded-full transition-all duration-500 ${
                                  cName === 'healthy'
                                    ? 'bg-emerald-500'
                                    : cName === 'bacterial'
                                    ? 'bg-rose-500'
                                    : cName === 'fungi'
                                    ? 'bg-amber-500'
                                    : 'bg-indigo-500'
                                }`}
                                style={{ width: `${pct}%` }}
                              />
                            </div>
                          </div>
                        );
                      })}
                    </div>

                    {/* Uncertainty Metrics */}
                    <div className="mt-5 pt-4 border-t border-slate-100 grid grid-cols-3 gap-2 text-center text-xs">
                      <div className="bg-slate-50 p-2 rounded-lg border border-slate-200">
                        <span className="text-slate-400 block text-[10px] font-semibold uppercase">Margin</span>
                        <span className="font-mono font-bold text-slate-800">
                          {(diagnosisResult.gate.margin * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="bg-slate-50 p-2 rounded-lg border border-slate-200">
                        <span className="text-slate-400 block text-[10px] font-semibold uppercase">Normalized Entropy</span>
                        <span className="font-mono font-bold text-slate-800">
                          {diagnosisResult.gate.entropy.toFixed(3)}
                        </span>
                      </div>
                      <div className="bg-slate-50 p-2 rounded-lg border border-slate-200">
                        <span className="text-slate-400 block text-[10px] font-semibold uppercase">Latency</span>
                        <span className="font-mono font-bold text-emerald-700">
                          {diagnosisResult.latency_ms} ms
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Telemetry Footnote */}
                  <div className="text-[11px] text-slate-400 flex flex-wrap justify-between items-center px-1">
                    <span>Backbone: <strong className="text-slate-600">{diagnosisResult.model.backbone}</strong></span>
                    <span>Metric: <strong className="text-slate-600">{diagnosisResult.model.metric}</strong></span>
                    <span>Request ID: <code className="bg-slate-100 px-1 py-0.5 rounded text-slate-600">{diagnosisResult.request_id}</code></span>
                    <span>Time: {new Date(diagnosisResult.timestamp).toLocaleTimeString()}</span>
                  </div>
                </div>
              ) : (
                <div className="bg-white rounded-xl border border-dashed border-slate-300 p-12 text-center text-slate-400 flex flex-col items-center justify-center min-h-[460px]">
                  <div className="w-16 h-16 rounded-full bg-slate-100 flex items-center justify-center mb-3 text-slate-400">
                    <Icons.Leaf />
                  </div>
                  <h3 className="text-sm font-semibold text-slate-600">No Specimen Diagnosed</h3>
                  <p className="text-xs text-slate-400 max-w-sm mt-1">
                    Upload a foliar photo on the left and run diagnosis to view real-time disease classification, confidence gating, and export quarantine protocols.
                  </p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ========================================================================= */}
        {/* TAB 2: AGRONOMIST REVIEW QUEUE                                           */}
        {/* ========================================================================= */}
        {activeTab === 'review' && (
          <div className="space-y-6">
            <div className="bg-white rounded-xl p-5 border border-slate-200 shadow-xs flex flex-col md:flex-row md:items-center justify-between gap-4">
              <div>
                <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
                  <Icons.ClipboardList /> Human-in-the-Loop Review Queue
                </h2>
                <p className="text-xs text-slate-500 mt-0.5">
                  Foliar cases flagged as <code>UNKNOWN_ANOMALY</code> quarantined for expert agronomist validation.
                </p>
              </div>

              {/* Status Filter Tabs */}
              <div className="flex items-center gap-2 bg-slate-100 p-1 rounded-lg text-xs font-medium">
                {['ALL', 'PENDING', 'RESOLVED'].map(st => (
                  <button
                    key={st}
                    onClick={() => setReviewFilter(st)}
                    className={`px-3 py-1.5 rounded-md transition ${
                      reviewFilter === st
                        ? 'bg-white text-emerald-800 font-bold shadow-xs'
                        : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    {st} {st === 'PENDING' && reviewStats.pending > 0 && `(${reviewStats.pending})`}
                  </button>
                ))}
              </div>
            </div>

            {/* Ticket Table */}
            <div className="bg-white rounded-xl border border-slate-200 shadow-xs overflow-hidden">
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="bg-slate-50 border-b border-slate-200 text-slate-600 uppercase font-semibold">
                    <tr>
                      <th className="py-3 px-4">Ticket ID</th>
                      <th className="py-3 px-4">Captured</th>
                      <th className="py-3 px-4">Plant & Species</th>
                      <th className="py-3 px-4">Model Prediction</th>
                      <th className="py-3 px-4">Confidence</th>
                      <th className="py-3 px-4">Status</th>
                      <th className="py-3 px-4 text-right">Action</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {filteredTickets.length > 0 ? (
                      filteredTickets.map(ticket => (
                        <tr key={ticket.ticket_id} className="hover:bg-slate-50/80 transition">
                          <td className="py-3 px-4 font-mono font-bold text-slate-700">
                            #{ticket.ticket_id}
                          </td>
                          <td className="py-3 px-4 text-slate-500">
                            {new Date(ticket.created_at).toLocaleString()}
                          </td>
                          <td className="py-3 px-4">
                            <span className="font-semibold text-slate-800">{ticket.species || 'Unknown Plant'}</span>
                            <span className="block text-[10px] text-slate-400 font-mono">{ticket.plant_id || 'No Tag'}</span>
                          </td>
                          <td className="py-3 px-4">
                            <span className="capitalize font-semibold text-slate-700">{ticket.top_prediction}</span>
                          </td>
                          <td className="py-3 px-4 font-mono text-slate-600">
                            {(ticket.confidence * 100).toFixed(1)}%
                          </td>
                          <td className="py-3 px-4">
                            <span className={`px-2 py-0.5 rounded text-[10px] font-bold tracking-wide uppercase ${
                              ticket.status === 'PENDING'
                                ? 'bg-amber-100 text-amber-800 border border-amber-200'
                                : 'bg-emerald-100 text-emerald-800 border border-emerald-200'
                            }`}>
                              {ticket.status}
                            </span>
                          </td>
                          <td className="py-3 px-4 text-right">
                            {ticket.status === 'PENDING' ? (
                              <button
                                onClick={() => setSelectedTicket(ticket)}
                                className="px-3 py-1 bg-emerald-700 hover:bg-emerald-800 text-white rounded font-medium text-xs shadow-xs"
                              >
                                Triage & Resolve
                              </button>
                            ) : (
                              <span className="text-[11px] text-slate-400 italic">
                                Resolved as <strong className="capitalize text-slate-600">{ticket.resolution?.label}</strong>
                              </span>
                            )}
                          </td>
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td colSpan={7} className="py-8 text-center text-slate-400">
                          No triage tickets found.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Resolve Modal Dialog */}
            {selectedTicket && (
              <div className="fixed inset-0 bg-slate-900/40 backdrop-blur-xs flex items-center justify-center p-4 z-50">
                <div className="bg-white rounded-2xl max-w-lg w-full p-6 shadow-2xl border border-slate-200">
                  <div className="flex justify-between items-start mb-4">
                    <div>
                      <h3 className="text-base font-bold text-slate-900">Resolve Review Ticket</h3>
                      <p className="text-xs text-slate-500 font-mono">ID: #{selectedTicket.ticket_id}</p>
                    </div>
                    <button
                      onClick={() => setSelectedTicket(null)}
                      className="text-slate-400 hover:text-slate-600 text-lg font-bold"
                    >
                      &times;
                    </button>
                  </div>

                  <div className="space-y-4 text-xs">
                    <div className="bg-slate-50 p-3 rounded-lg border border-slate-200 flex justify-between">
                      <div>
                        <span className="text-slate-500 block text-[10px] uppercase font-bold">Prediction</span>
                        <span className="capitalize font-semibold text-slate-800">{selectedTicket.top_prediction}</span>
                      </div>
                      <div>
                        <span className="text-slate-500 block text-[10px] uppercase font-bold">Confidence</span>
                        <span className="font-mono font-semibold text-slate-800">{(selectedTicket.confidence * 100).toFixed(1)}%</span>
                      </div>
                      <div>
                        <span className="text-slate-500 block text-[10px] uppercase font-bold">Species</span>
                        <span className="font-semibold text-slate-800">{selectedTicket.species}</span>
                      </div>
                    </div>

                    <div>
                      <label className="block font-medium text-slate-700 mb-1">Validated Diagnosis Label</label>
                      <select
                        value={resolveLabel}
                        onChange={(e) => setResolveLabel(e.target.value)}
                        className="w-full px-3 py-2 border border-slate-300 rounded-lg text-xs focus:ring-2 focus:ring-emerald-500 focus:outline-hidden"
                      >
                        <option value="healthy">healthy</option>
                        <option value="bacterial">bacterial</option>
                        <option value="fungi">fungi</option>
                        <option value="abiotic">abiotic</option>
                        <option value="discard">discard</option>
                      </select>
                    </div>

                    <div>
                      <label className="block font-medium text-slate-700 mb-1">Reviewer Name</label>
                      <input
                        type="text"
                        value={reviewerName}
                        onChange={(e) => setReviewerName(e.target.value)}
                        className="w-full px-3 py-2 border border-slate-300 rounded-lg text-xs focus:ring-2 focus:ring-emerald-500 focus:outline-hidden"
                      />
                    </div>

                    <div>
                      <label className="block font-medium text-slate-700 mb-1">Agronomic Notes</label>
                      <textarea
                        rows={2}
                        value={resolveNotes}
                        onChange={(e) => setResolveNotes(e.target.value)}
                        className="w-full px-3 py-2 border border-slate-300 rounded-lg text-xs focus:ring-2 focus:ring-emerald-500 focus:outline-hidden"
                      />
                    </div>

                    <div className="flex items-center gap-2 p-3 bg-emerald-50/60 border border-emerald-200 rounded-lg">
                      <input
                        type="checkbox"
                        id="add-proto"
                        checked={addToProto}
                        onChange={(e) => setAddToProto(e.target.checked)}
                        className="rounded text-emerald-600 focus:ring-emerald-500"
                      />
                      <label htmlFor="add-proto" className="text-emerald-900 font-semibold cursor-pointer">
                        Add to Prototype Support Set
                      </label>
                    </div>

                    <div className="flex justify-end gap-2 pt-2 border-t border-slate-100">
                      <button
                        onClick={() => setSelectedTicket(null)}
                        className="px-4 py-2 border border-slate-300 rounded-lg text-slate-700 hover:bg-slate-50 font-medium"
                      >
                        Cancel
                      </button>
                      <button
                        onClick={handleResolveTicket}
                        disabled={isResolving}
                        className="px-4 py-2 bg-emerald-700 hover:bg-emerald-800 text-white rounded-lg font-semibold shadow-xs"
                      >
                        {isResolving ? 'Submitting...' : 'Confirm Resolution'}
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ========================================================================= */}
        {/* TAB 3: DISEASE PROTOTYPES (FEW-SHOT STORE)                               */}
        {/* ========================================================================= */}
        {activeTab === 'prototypes' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
            {/* Active Class Centroids */}
            <div className="lg:col-span-7 space-y-4">
              <div className="bg-white rounded-xl p-5 border border-slate-200 shadow-xs">
                <div className="flex justify-between items-center mb-4">
                  <div>
                    <h2 className="text-base font-bold text-slate-900">Registered Disease Prototypes</h2>
                    <p className="text-xs text-slate-500">Active class centroid representations</p>
                  </div>
                  <button
                    onClick={fetchPrototypes}
                    className="text-xs px-2.5 py-1.5 border border-slate-300 rounded-lg hover:bg-slate-50 text-slate-700 font-medium"
                  >
                    {protoLoading ? 'Refreshing...' : 'Refresh Store'}
                  </button>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  {prototypes.map((proto) => (
                    <div
                      key={proto.name}
                      className="p-4 rounded-xl border border-slate-200 bg-slate-50/50 hover:bg-white hover:border-emerald-300 transition"
                    >
                      <div className="flex justify-between items-start mb-2">
                        <div>
                          <span className="font-bold text-slate-900 text-sm capitalize">{proto.display_name || proto.name}</span>
                          <span className="block text-[11px] font-mono text-slate-400">{proto.name}</span>
                        </div>
                        <span className={`text-[10px] font-bold uppercase px-2 py-0.5 rounded ${
                          proto.kind === 'core' ? 'bg-blue-100 text-blue-800' : 'bg-purple-100 text-purple-800'
                        }`}>
                          {proto.kind}
                        </span>
                      </div>
                      <div className="text-xs text-slate-600 flex justify-between items-center pt-2 border-t border-slate-200/60 mt-2">
                        <span>Support Images: <strong className="text-slate-800">{proto.n_support}</strong></span>
                        <span className="text-[10px] text-slate-400">128-D Normalized</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Incremental Support Addition Box */}
              <div className="bg-white rounded-xl p-5 border border-slate-200 shadow-xs">
                <h3 className="text-sm font-bold text-slate-900 mb-1">Add Support Images to Existing Class</h3>
                <p className="text-xs text-slate-500 mb-3">
                  Updates the prototype centroid via running mean without retraining.
                </p>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-3 text-xs">
                  <div>
                    <label className="block font-medium text-slate-700 mb-1">Target Class</label>
                    <select
                      value={supportClassName}
                      onChange={(e) => setSupportClassName(e.target.value)}
                      className="w-full px-3 py-2 border border-slate-300 rounded-lg"
                    >
                      <option value="">Select prototype class...</option>
                      {prototypes.map(p => (
                        <option key={p.name} value={p.name}>{p.display_name} ({p.name})</option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="block font-medium text-slate-700 mb-1">Upload Additional Photos</label>
                    <input
                      type="file"
                      multiple
                      onChange={(e) => setSupportFiles(e.target.files)}
                      className="w-full text-xs text-slate-500 file:mr-2 file:py-1.5 file:px-3 file:rounded-md file:border-0 file:text-xs file:font-semibold file:bg-emerald-50 file:text-emerald-700 hover:file:bg-emerald-100"
                    />
                  </div>
                </div>
                <button
                  disabled={!supportClassName || supportFiles.length === 0 || isAddingSupport}
                  onClick={async () => {
                    setIsAddingSupport(true);
                    try {
                      const formData = new FormData();
                      Array.from(supportFiles).forEach(f => formData.append('files', f));
                      const res = await fetch(`${apiBase}/prototypes/${supportClassName}/support`, {
                        method: 'POST',
                        headers: { 'X-API-Key': adminApiKey },
                        body: formData
                      });
                      if (res.ok) {
                        fetchPrototypes();
                        alert(`Successfully added support images to ${supportClassName}`);
                        setSupportFiles([]);
                      } else {
                        throw new Error(`HTTP ${res.status}`);
                      }
                    } catch (err) {
                      alert(err.message || "Failed to update class prototype.");
                    } finally {
                      setIsAddingSupport(false);
                    }
                  }}
                  className={`w-full py-2 rounded-lg font-semibold text-xs transition ${
                    !supportClassName || supportFiles.length === 0
                      ? 'bg-slate-200 text-slate-400 cursor-not-allowed'
                      : 'bg-slate-800 hover:bg-slate-900 text-white'
                  }`}
                >
                  {isAddingSupport ? 'Updating Centroid...' : 'Update Class Prototype'}
                </button>
              </div>
            </div>

            {/* Dynamic Class Enrollment Form */}
            <div className="lg:col-span-5 bg-white rounded-xl p-5 border border-slate-200 shadow-xs">
              <h2 className="text-base font-bold text-slate-900 mb-1">Enroll New Condition</h2>
              <p className="text-xs text-slate-500 mb-4 leading-relaxed">
                Add an emerging disease class dynamically by providing labeled reference photos.
              </p>

              <form onSubmit={handleRegisterDisease} className="space-y-3 text-xs">
                <div>
                  <label className="block font-medium text-slate-700 mb-1">Class Identifier (lowercase, e.g. anthracnose)</label>
                  <input
                    type="text"
                    required
                    value={newClassName}
                    onChange={(e) => setNewClassName(e.target.value)}
                    placeholder="e.g. anthracnose"
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:outline-hidden font-mono"
                  />
                </div>

                <div>
                  <label className="block font-medium text-slate-700 mb-1">Display Name</label>
                  <input
                    type="text"
                    value={newDisplayName}
                    onChange={(e) => setNewDisplayName(e.target.value)}
                    placeholder="e.g. Anthracnose Spot"
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:outline-hidden"
                  />
                </div>

                <div>
                  <label className="block font-medium text-slate-700 mb-1">Support Images</label>
                  <input
                    type="file"
                    multiple
                    required
                    onChange={(e) => setEnrollFiles(e.target.files)}
                    className="w-full text-xs text-slate-500 file:mr-2 file:py-1.5 file:px-3 file:rounded-md file:border-0 file:text-xs file:font-semibold file:bg-emerald-50 file:text-emerald-700 hover:file:bg-emerald-100"
                  />
                </div>

                <button
                  type="submit"
                  disabled={isEnrolling}
                  className="w-full mt-2 py-2.5 bg-emerald-700 hover:bg-emerald-800 text-white font-semibold rounded-lg shadow-xs transition"
                >
                  {isEnrolling ? 'Enrolling Condition...' : 'Enroll Disease Class'}
                </button>
              </form>

              {/* Enrollment Result Card */}
              {enrollResult && (
                <div className="mt-4 p-4 rounded-xl bg-slate-50 border border-slate-200 text-xs space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="font-bold text-slate-800">Enrollment Result</span>
                    <span className="text-emerald-700 font-semibold">&check; Registered</span>
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-[11px] pt-1">
                    <div>
                      <span className="text-slate-400 block">Intra-Class Cohesion:</span>
                      <strong className="text-slate-800 font-mono">{(enrollResult.data.cohesion * 100).toFixed(1)}%</strong>
                    </div>
                    <div>
                      <span className="text-slate-400 block">Nearest Prototype:</span>
                      <strong className="text-slate-800 capitalize">{enrollResult.data.nearest_class}</strong>
                    </div>
                  </div>
                  {enrollResult.data.warnings && enrollResult.data.warnings.length > 0 && (
                    <div className="text-[11px] text-amber-700 bg-amber-50 p-2 rounded border border-amber-200">
                      {enrollResult.data.warnings.join(' ')}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}