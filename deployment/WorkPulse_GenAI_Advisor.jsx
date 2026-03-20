import { useState, useRef, useEffect } from "react";

// ── WorkPulse XGBoost Model (simplified JS implementation) ──
// Replicates the key decision logic from our trained model
function predictBurnoutRisk(employee) {
  const {
    overtime_index, wellbeing_composite, workload_pressure,
    satisfaction_gap, high_stress_flag, tenure_risk_flag,
    job_satisfaction, work_life_balance, tenure_years, age
  } = employee;

  // Non-linear burnout score (matches our training data generation)
  const ot = overtime_index;
  const wb = wellbeing_composite;
  const sf = high_stress_flag;
  const js = job_satisfaction;
  const wp = workload_pressure;
  const tr = tenure_risk_flag;
  const sg = satisfaction_gap;
  const ten = tenure_years;

  const score =
    0.18 * (ot > 0.35 ? (ot - 0.35) ** 2 * 10 + 0.3 * ot : ot * 0.5) +
    0.22 * sf * (1 - wb) ** 1.5 +
    0.12 * wp * (1 - js) +
    0.08 * (1 - wb) +
    0.06 * sf +
    0.05 * tr +
    0.06 * (Math.exp(-0.5 * ((ten - 2) / 1.5) ** 2) + 0.4 * Math.exp(-0.5 * ((ten - 17) / 4) ** 2)) +
    0.04 * Math.tanh(-sg * 2.5) +
    0.05 * ot * (age < 35 ? 1.4 : 0.7) -
    0.06 * sf * js;

  // Calibrate to probability
  const prob = Math.min(0.99, Math.max(0.01, score * 2.8 - 0.15));
  const risk = prob >= 0.5 ? 1 : 0;
  const level = prob < 0.3 ? "Low" : prob < 0.6 ? "Medium" : "High";

  // Top contributing factors
  const factors = [
    { feature: "high_stress_flag", value: sf, impact: 0.22 * sf * (1 - wb) ** 1.5 + 0.06 * sf },
    { feature: "overtime_index", value: ot, impact: 0.18 * (ot > 0.35 ? (ot - 0.35) ** 2 * 10 : ot * 0.5) },
    { feature: "wellbeing_composite", value: wb, impact: 0.08 * (1 - wb) },
    { feature: "tenure_risk_flag", value: tr, impact: 0.05 * tr },
    { feature: "workload_pressure", value: wp, impact: 0.12 * wp * (1 - js) },
    { feature: "job_satisfaction", value: js, impact: 0.12 * wp * (1 - js) },
  ].sort((a, b) => b.impact - a.impact);

  return { probability: prob, risk, level, factors, score };
}

// ── Sample employee profiles ──
const PRESETS = {
  high_risk: {
    name: "Alex Chen", role: "Senior Developer", department: "Engineering",
    overtime_index: 0.78, wellbeing_composite: 0.22, workload_pressure: 0.72,
    satisfaction_gap: -0.35, high_stress_flag: 1, tenure_risk_flag: 1,
    job_satisfaction: 0.28, work_life_balance: 0.18, log_income: 7.8,
    monthly_income: 4500, tenure_years: 2.5, age: 28, age_group: 0,
  },
  medium_risk: {
    name: "Sarah Kim", role: "Product Manager", department: "Product",
    overtime_index: 0.42, wellbeing_composite: 0.52, workload_pressure: 0.38,
    satisfaction_gap: -0.1, high_stress_flag: 1, tenure_risk_flag: 0,
    job_satisfaction: 0.55, work_life_balance: 0.50, log_income: 8.8,
    monthly_income: 8000, tenure_years: 4, age: 34, age_group: 1,
  },
  low_risk: {
    name: "James Miller", role: "Staff Engineer", department: "Engineering",
    overtime_index: 0.12, wellbeing_composite: 0.82, workload_pressure: 0.10,
    satisfaction_gap: 0.15, high_stress_flag: 0, tenure_risk_flag: 0,
    job_satisfaction: 0.82, work_life_balance: 0.88, log_income: 9.8,
    monthly_income: 15000, tenure_years: 8, age: 42, age_group: 2,
  },
};

const RISK_COLORS = { High: "#ef4444", Medium: "#f59e0b", Low: "#22c55e" };
const RISK_BG = { High: "#fef2f2", Medium: "#fffbeb", Low: "#f0fdf4" };

export default function WorkPulseAdvisor() {
  const [employee, setEmployee] = useState(PRESETS.high_risk);
  const [prediction, setPrediction] = useState(null);
  const [aiInsight, setAiInsight] = useState("");
  const [aiLoading, setAiLoading] = useState(false);
  const [edaSummary, setEdaSummary] = useState("");
  const [edaLoading, setEdaLoading] = useState(false);
  const [activeTab, setActiveTab] = useState("assess");
  const [chatMessages, setChatMessages] = useState([]);
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const chatEndRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatMessages]);

  // Run prediction
  const runPrediction = () => {
    const result = predictBurnoutRisk(employee);
    setPrediction(result);
    setAiInsight("");
    return result;
  };

  // Call Claude API for AI insights
  const generateAIInsight = async (pred) => {
    setAiLoading(true);
    try {
      const response = await fetch("https://api.anthropic.com/v1/messages", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "claude-sonnet-4-20250514",
          max_tokens: 1000,
          messages: [{
            role: "user",
            content: `You are WorkPulse, an AI-powered employee burnout advisor for HR managers. Analyze this employee's burnout risk assessment and provide actionable guidance.

Employee: ${employee.name}, ${employee.role} in ${employee.department}
Age: ${employee.age} | Tenure: ${employee.tenure_years} years

Risk Assessment:
- Burnout Probability: ${(pred.probability * 100).toFixed(1)}%
- Risk Level: ${pred.level}
- Top Risk Factors: ${pred.factors.slice(0, 3).map(f => `${f.feature} (${f.value.toFixed(2)})`).join(", ")}

Key Metrics:
- Overtime Index: ${employee.overtime_index.toFixed(2)} (0=none, 1=max)
- Wellbeing: ${employee.wellbeing_composite.toFixed(2)} (0=low, 1=high)
- Stress Flag: ${employee.high_stress_flag ? "HIGH" : "Normal"}
- Job Satisfaction: ${employee.job_satisfaction.toFixed(2)}
- Work-Life Balance: ${employee.work_life_balance.toFixed(2)}

Provide:
1. A 2-sentence risk summary explaining WHY this employee is at ${pred.level} risk
2. The #1 most urgent action item for their manager
3. Two specific, compassionate intervention suggestions tailored to their profile
4. One thing NOT to do (common HR mistake)

Keep it concise, warm, and actionable. Use plain language suitable for a non-technical HR manager. Do not use markdown headers — use plain text with line breaks.`
          }],
        }),
      });
      const data = await response.json();
      const text = data.content?.map(c => c.text || "").join("\n") || "Unable to generate insight.";
      setAiInsight(text);
    } catch (err) {
      setAiInsight("AI insight generation failed. The prediction model still works independently — this is an optional enhancement.");
    }
    setAiLoading(false);
  };

  // Generate EDA summary
  const generateEDASummary = async () => {
    setEdaLoading(true);
    try {
      const response = await fetch("https://api.anthropic.com/v1/messages", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "claude-sonnet-4-20250514",
          max_tokens: 1000,
          messages: [{
            role: "user",
            content: `You are a data scientist generating an automated EDA summary for the WorkPulse burnout prediction dataset. The dataset has 44,220 employee records with 13 features.

Dataset statistics:
- 35% of employees flagged as high burnout risk (target imbalance: 65/35)
- Features: overtime_index (mean=0.30, std=0.18), wellbeing_composite (mean=0.55, std=0.15), high_stress_flag (38% flagged), tenure_risk_flag (42% in risk windows), job_satisfaction (mean=0.50, std=0.20), work_life_balance (mean=0.55, std=0.18)
- Non-linear patterns: overtime threshold effect at 0.35, U-shaped tenure risk (peaks at 2yr and 17yr), multiplicative stress × wellbeing interaction

Generate a concise automated EDA summary covering:
1. Dataset overview (2 sentences)
2. Key distribution insights (3 findings)
3. Correlation highlights (2 findings)
4. Feature interaction discoveries (2 findings)
5. Data quality notes (1-2 sentences)

Write as a professional data science report paragraph. No markdown headers — use flowing prose with numbered points where helpful.`
          }],
        }),
      });
      const data = await response.json();
      setEdaSummary(data.content?.map(c => c.text || "").join("\n") || "Unable to generate summary.");
    } catch (err) {
      setEdaSummary("EDA summary generation failed. Check API connectivity.");
    }
    setEdaLoading(false);
  };

  // Chat with WorkPulse AI
  const sendChatMessage = async () => {
    if (!chatInput.trim()) return;
    const userMsg = chatInput.trim();
    setChatInput("");
    setChatMessages(prev => [...prev, { role: "user", text: userMsg }]);
    setChatLoading(true);

    const context = prediction
      ? `Current employee: ${employee.name}, ${employee.role}. Risk: ${prediction.level} (${(prediction.probability * 100).toFixed(1)}%). Overtime: ${employee.overtime_index.toFixed(2)}, Wellbeing: ${employee.wellbeing_composite.toFixed(2)}, Stress: ${employee.high_stress_flag ? "HIGH" : "Normal"}, Satisfaction: ${employee.job_satisfaction.toFixed(2)}.`
      : "No employee currently assessed.";

    try {
      const history = chatMessages.slice(-6).map(m => ({
        role: m.role === "user" ? "user" : "assistant",
        content: m.text,
      }));

      const response = await fetch("https://api.anthropic.com/v1/messages", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "claude-sonnet-4-20250514",
          max_tokens: 600,
          system: `You are WorkPulse AI, a burnout prevention advisor for HR managers. You help interpret employee risk scores, suggest interventions, and answer questions about the WorkPulse ML model (XGBoost, F1=0.91, AUC=0.99, trained on 44K employees). Be concise, warm, and practical. Current context: ${context}`,
          messages: [...history, { role: "user", content: userMsg }],
        }),
      });
      const data = await response.json();
      const reply = data.content?.map(c => c.text || "").join("\n") || "I couldn't process that request.";
      setChatMessages(prev => [...prev, { role: "assistant", text: reply }]);
    } catch {
      setChatMessages(prev => [...prev, { role: "assistant", text: "Connection error. Please try again." }]);
    }
    setChatLoading(false);
  };

  const selectPreset = (key) => {
    setEmployee(PRESETS[key]);
    setPrediction(null);
    setAiInsight("");
  };

  const handlePredict = () => {
    const result = runPrediction();
    generateAIInsight(result);
  };

  // ── Render ──
  return (
    <div style={{
      fontFamily: "'DM Sans', 'Segoe UI', sans-serif",
      background: "linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%)",
      minHeight: "100vh", color: "#e2e8f0", padding: 0, margin: 0,
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
        * { box-sizing: border-box; }
        .glow { box-shadow: 0 0 30px rgba(13,148,136,0.15); }
        .tab-btn { padding: 10px 20px; border: none; cursor: pointer; font-size: 13px; font-weight: 600;
          font-family: 'DM Sans', sans-serif; border-radius: 8px 8px 0 0; transition: all 0.2s; }
        .tab-active { background: #1e293b; color: #0d9488; border-bottom: 2px solid #0d9488; }
        .tab-inactive { background: #0f172a; color: #64748b; }
        .tab-inactive:hover { color: #94a3b8; }
        .preset-btn { padding: 8px 16px; border: 1px solid #334155; border-radius: 8px; cursor: pointer;
          font-size: 12px; font-family: 'DM Sans', sans-serif; transition: all 0.2s; background: #0f172a; color: #94a3b8; }
        .preset-btn:hover { border-color: #0d9488; color: #0d9488; }
        .preset-active { border-color: #0d9488; color: #0d9488; background: rgba(13,148,136,0.1); }
        .slider-container { margin-bottom: 14px; }
        .slider-label { display: flex; justify-content: space-between; font-size: 12px; color: #94a3b8; margin-bottom: 4px; }
        .slider-val { color: #0d9488; font-weight: 600; font-family: 'JetBrains Mono', monospace; }
        input[type="range"] { width: 100%; height: 6px; -webkit-appearance: none; background: #334155;
          border-radius: 3px; outline: none; }
        input[type="range"]::-webkit-slider-thumb { -webkit-appearance: none; width: 16px; height: 16px;
          background: #0d9488; border-radius: 50%; cursor: pointer; }
        .predict-btn { width: 100%; padding: 14px; background: linear-gradient(135deg, #0d9488, #0f766e);
          color: white; border: none; border-radius: 10px; font-size: 15px; font-weight: 700;
          cursor: pointer; font-family: 'DM Sans', sans-serif; letter-spacing: 0.5px; transition: all 0.3s; }
        .predict-btn:hover { transform: translateY(-1px); box-shadow: 0 4px 20px rgba(13,148,136,0.4); }
        .risk-gauge { position: relative; width: 180px; height: 180px; margin: 0 auto; }
        .chat-input { flex: 1; padding: 12px 16px; background: #0f172a; border: 1px solid #334155;
          border-radius: 10px; color: #e2e8f0; font-size: 14px; font-family: 'DM Sans', sans-serif; outline: none; }
        .chat-input:focus { border-color: #0d9488; }
        .chat-send { padding: 12px 20px; background: #0d9488; color: white; border: none; border-radius: 10px;
          font-weight: 600; cursor: pointer; font-family: 'DM Sans', sans-serif; }
        .chat-msg { padding: 12px 16px; border-radius: 12px; max-width: 85%; font-size: 14px; line-height: 1.5; white-space: pre-wrap; }
        .chat-user { background: #0d9488; color: white; margin-left: auto; }
        .chat-ai { background: #1e293b; color: #e2e8f0; border: 1px solid #334155; }
        .pulse { animation: pulse 2s infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
      `}</style>

      {/* Header */}
      <div style={{ padding: "20px 28px", borderBottom: "1px solid #1e293b", display: "flex", alignItems: "center", gap: 12 }}>
        <div style={{ width: 36, height: 36, borderRadius: 10, background: "linear-gradient(135deg, #0d9488, #f97316)",
          display: "flex", alignItems: "center", justifyContent: "center", fontSize: 18 }}>🔥</div>
        <div>
          <div style={{ fontSize: 18, fontWeight: 700, color: "#f8fafc", letterSpacing: 1 }}>WORKPULSE AI</div>
          <div style={{ fontSize: 11, color: "#64748b", letterSpacing: 0.5 }}>GenAI-Enhanced Burnout Advisor • Step 9 Demo</div>
        </div>
        <div style={{ marginLeft: "auto", fontSize: 11, color: "#475569", textAlign: "right" }}>
          <div>Model: XGBoost (Tuned)</div>
          <div>F1=0.91 | AUC=0.99 | Powered by Claude API</div>
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: "flex", gap: 2, padding: "0 28px", marginTop: 16 }}>
        {[
          { key: "assess", label: "🎯 Risk Assessment" },
          { key: "chat", label: "💬 AI Advisor Chat" },
          { key: "eda", label: "📊 Auto-EDA Summary" },
        ].map(t => (
          <button key={t.key} className={`tab-btn ${activeTab === t.key ? "tab-active" : "tab-inactive"}`}
            onClick={() => setActiveTab(t.key)}>{t.label}</button>
        ))}
      </div>

      <div style={{ padding: "0 28px 28px" }}>
        {/* ══════ TAB: RISK ASSESSMENT ══════ */}
        {activeTab === "assess" && (
          <div style={{ background: "#1e293b", borderRadius: "0 12px 12px 12px", padding: 24 }} className="glow">
            {/* Presets */}
            <div style={{ display: "flex", gap: 8, marginBottom: 20, flexWrap: "wrap" }}>
              <span style={{ fontSize: 12, color: "#64748b", alignSelf: "center", marginRight: 4 }}>Sample profiles:</span>
              {Object.entries(PRESETS).map(([key, p]) => (
                <button key={key} className={`preset-btn ${employee.name === p.name ? "preset-active" : ""}`}
                  onClick={() => selectPreset(key)}>
                  {p.name} ({key.replace("_", " ")})
                </button>
              ))}
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
              {/* Left: Input sliders */}
              <div>
                <div style={{ fontSize: 14, fontWeight: 600, color: "#94a3b8", marginBottom: 12, letterSpacing: 0.5 }}>EMPLOYEE PROFILE</div>
                {[
                  { key: "overtime_index", label: "Overtime Index", min: 0, max: 1, step: 0.01 },
                  { key: "wellbeing_composite", label: "Wellbeing Composite", min: 0, max: 1, step: 0.01 },
                  { key: "workload_pressure", label: "Workload Pressure", min: 0, max: 1, step: 0.01 },
                  { key: "job_satisfaction", label: "Job Satisfaction", min: 0, max: 1, step: 0.01 },
                  { key: "work_life_balance", label: "Work-Life Balance", min: 0, max: 1, step: 0.01 },
                  { key: "satisfaction_gap", label: "Satisfaction Gap", min: -1, max: 1, step: 0.01 },
                  { key: "tenure_years", label: "Tenure (years)", min: 0, max: 40, step: 0.5 },
                  { key: "age", label: "Age", min: 22, max: 60, step: 1 },
                ].map(s => (
                  <div key={s.key} className="slider-container">
                    <div className="slider-label">
                      <span>{s.label}</span>
                      <span className="slider-val">{employee[s.key]?.toFixed(s.step < 1 ? 2 : 0)}</span>
                    </div>
                    <input type="range" min={s.min} max={s.max} step={s.step}
                      value={employee[s.key] || 0}
                      onChange={e => setEmployee({ ...employee, [s.key]: parseFloat(e.target.value) })} />
                  </div>
                ))}
                <div style={{ display: "flex", gap: 12, marginBottom: 16 }}>
                  {[{ key: "high_stress_flag", label: "High Stress" }, { key: "tenure_risk_flag", label: "Tenure Risk Window" }].map(t => (
                    <label key={t.key} style={{ display: "flex", alignItems: "center", gap: 6, cursor: "pointer", fontSize: 13 }}>
                      <input type="checkbox" checked={!!employee[t.key]}
                        onChange={e => setEmployee({ ...employee, [t.key]: e.target.checked ? 1 : 0 })}
                        style={{ accentColor: "#0d9488" }} />
                      {t.label}
                    </label>
                  ))}
                </div>
                <button className="predict-btn" onClick={handlePredict}>
                  🔍 Assess Burnout Risk + Generate AI Insight
                </button>
              </div>

              {/* Right: Results */}
              <div>
                {prediction ? (
                  <div>
                    {/* Risk gauge */}
                    <div style={{ textAlign: "center", marginBottom: 16 }}>
                      <div style={{ fontSize: 13, color: "#64748b", marginBottom: 8, fontWeight: 600, letterSpacing: 0.5 }}>BURNOUT RISK</div>
                      <div style={{
                        width: 140, height: 140, borderRadius: "50%", margin: "0 auto",
                        background: `conic-gradient(${RISK_COLORS[prediction.level]} ${prediction.probability * 360}deg, #1e293b ${prediction.probability * 360}deg)`,
                        display: "flex", alignItems: "center", justifyContent: "center",
                        boxShadow: `0 0 30px ${RISK_COLORS[prediction.level]}40`,
                      }}>
                        <div style={{
                          width: 110, height: 110, borderRadius: "50%", background: "#0f172a",
                          display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
                        }}>
                          <div style={{ fontSize: 32, fontWeight: 700, color: RISK_COLORS[prediction.level],
                            fontFamily: "'JetBrains Mono', monospace" }}>
                            {(prediction.probability * 100).toFixed(0)}%
                          </div>
                          <div style={{ fontSize: 13, fontWeight: 600, color: RISK_COLORS[prediction.level] }}>{prediction.level} Risk</div>
                        </div>
                      </div>
                    </div>

                    {/* Top factors */}
                    <div style={{ fontSize: 13, color: "#64748b", fontWeight: 600, marginBottom: 8, letterSpacing: 0.5 }}>TOP RISK FACTORS</div>
                    {prediction.factors.slice(0, 4).map((f, i) => (
                      <div key={i} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                        <div style={{ flex: 1, fontSize: 12, color: "#94a3b8" }}>{f.feature.replace(/_/g, " ")}</div>
                        <div style={{ width: 80, height: 6, background: "#0f172a", borderRadius: 3, overflow: "hidden" }}>
                          <div style={{ width: `${Math.min(100, f.impact * 500)}%`, height: "100%",
                            background: `linear-gradient(90deg, #0d9488, ${RISK_COLORS[prediction.level]})`, borderRadius: 3 }} />
                        </div>
                        <div style={{ fontSize: 11, color: "#64748b", fontFamily: "'JetBrains Mono'", width: 36, textAlign: "right" }}>
                          {f.value.toFixed(2)}
                        </div>
                      </div>
                    ))}

                    {/* AI Insight */}
                    <div style={{ marginTop: 20, fontSize: 13, color: "#64748b", fontWeight: 600, marginBottom: 8, letterSpacing: 0.5 }}>
                      🤖 AI-GENERATED INSIGHT
                    </div>
                    <div style={{
                      background: RISK_BG[prediction.level] + "15", border: `1px solid ${RISK_COLORS[prediction.level]}30`,
                      borderRadius: 10, padding: 14, fontSize: 13, lineHeight: 1.6, color: "#cbd5e1",
                      minHeight: 80, whiteSpace: "pre-wrap",
                    }}>
                      {aiLoading ? (
                        <span className="pulse" style={{ color: "#0d9488" }}>Generating personalized insight with Claude AI...</span>
                      ) : aiInsight || "Click 'Assess Burnout Risk' to generate an AI-powered insight."}
                    </div>
                  </div>
                ) : (
                  <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
                    height: "100%", color: "#475569", textAlign: "center", gap: 12 }}>
                    <div style={{ fontSize: 48, opacity: 0.3 }}>🎯</div>
                    <div style={{ fontSize: 14 }}>Select a profile or adjust sliders, then click<br />"Assess Burnout Risk" to get started</div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* ══════ TAB: AI CHAT ══════ */}
        {activeTab === "chat" && (
          <div style={{ background: "#1e293b", borderRadius: "0 12px 12px 12px", padding: 24, height: 520, display: "flex", flexDirection: "column" }} className="glow">
            <div style={{ fontSize: 13, color: "#64748b", marginBottom: 12 }}>
              Chat with WorkPulse AI about burnout risk, interventions, model details, or HR strategy.
              {prediction && <span style={{ color: "#0d9488" }}> Current: {employee.name} ({prediction.level} risk)</span>}
            </div>

            <div style={{ flex: 1, overflowY: "auto", display: "flex", flexDirection: "column", gap: 10, marginBottom: 14, paddingRight: 8 }}>
              {chatMessages.length === 0 && (
                <div style={{ textAlign: "center", color: "#475569", marginTop: 60 }}>
                  <div style={{ fontSize: 36, marginBottom: 12, opacity: 0.3 }}>💬</div>
                  <div style={{ fontSize: 13 }}>Try asking:</div>
                  <div style={{ fontSize: 12, color: "#64748b", marginTop: 8, lineHeight: 2 }}>
                    {["What interventions work best for high overtime?",
                      "How does the model handle fairness across genders?",
                      "Explain the SHAP analysis in simple terms",
                      "Draft an email to a manager about this employee's risk"].map((q, i) => (
                      <div key={i} style={{ cursor: "pointer", color: "#0d9488", opacity: 0.7 }}
                        onClick={() => { setChatInput(q); }}>"{q}"</div>
                    ))}
                  </div>
                </div>
              )}
              {chatMessages.map((msg, i) => (
                <div key={i} style={{ display: "flex" }}>
                  <div className={`chat-msg ${msg.role === "user" ? "chat-user" : "chat-ai"}`}>{msg.text}</div>
                </div>
              ))}
              {chatLoading && (
                <div style={{ display: "flex" }}>
                  <div className="chat-msg chat-ai pulse">Thinking...</div>
                </div>
              )}
              <div ref={chatEndRef} />
            </div>

            <div style={{ display: "flex", gap: 8 }}>
              <input className="chat-input" value={chatInput} placeholder="Ask WorkPulse AI anything..."
                onChange={e => setChatInput(e.target.value)}
                onKeyDown={e => e.key === "Enter" && !chatLoading && sendChatMessage()} />
              <button className="chat-send" onClick={sendChatMessage} disabled={chatLoading}>Send</button>
            </div>
          </div>
        )}

        {/* ══════ TAB: AUTO-EDA ══════ */}
        {activeTab === "eda" && (
          <div style={{ background: "#1e293b", borderRadius: "0 12px 12px 12px", padding: 24 }} className="glow">
            <div style={{ fontSize: 14, fontWeight: 600, color: "#94a3b8", marginBottom: 8, letterSpacing: 0.5 }}>
              AUTO-GENERATED EDA SUMMARY
            </div>
            <div style={{ fontSize: 13, color: "#64748b", marginBottom: 16 }}>
              Uses Claude API to automatically generate an exploratory data analysis summary from dataset statistics.
              This demonstrates LLM-powered auto-documentation (Step 9 requirement).
            </div>

            <button className="predict-btn" onClick={generateEDASummary} disabled={edaLoading}
              style={{ marginBottom: 20, maxWidth: 320 }}>
              {edaLoading ? "Generating..." : "📊 Generate EDA Summary"}
            </button>

            {edaSummary && (
              <div style={{
                background: "#0f172a", border: "1px solid #334155", borderRadius: 12, padding: 20,
                fontSize: 14, lineHeight: 1.7, color: "#cbd5e1", whiteSpace: "pre-wrap",
              }}>
                {edaSummary}
              </div>
            )}

            <div style={{ marginTop: 24, padding: 16, background: "#0f172a", borderRadius: 10, border: "1px solid #334155" }}>
              <div style={{ fontSize: 13, fontWeight: 600, color: "#94a3b8", marginBottom: 8 }}>HOW THIS WORKS</div>
              <div style={{ fontSize: 12, color: "#64748b", lineHeight: 1.6 }}>
                1. Dataset statistics (means, distributions, correlations) are extracted from the training data{"\n"}
                2. These statistics are sent to Claude API as a structured prompt{"\n"}
                3. Claude generates a professional EDA narrative with key insights{"\n"}
                4. The summary covers distributions, correlations, interactions, and data quality{"\n\n"}
                This replaces hours of manual EDA documentation with a single API call, while maintaining
                the data scientist's ability to review and edit the output.
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <div style={{ textAlign: "center", padding: "16px 28px", fontSize: 11, color: "#334155", borderTop: "1px solid #1e293b" }}>
        WorkPulse Capstone • Step 9: GenAI-Enhanced Application • Powered by Claude API (Anthropic)
        • Model: XGBoost Tuned (F1=0.9062, AUC=0.9868)
      </div>
    </div>
  );
}
