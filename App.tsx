import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { NormalizedLandmark } from "@mediapipe/tasks-vision";
import Stage from "./components/Stage";
import {
  AnalyzingOverlay, AttentionList, ExerciseCard, HistoryPanel, IndexRing, MetricCard,
  ObservationList, Reveal, ScheduleStrip, SectionHead, Stepper, type HistoryEntry,
} from "./components/ReportPanels";
import {
  FigureFront, FigureSide, IconAlert, IconArrowRight, IconCamera, IconCameraHeight, IconCheck,
  IconCheckCircle, IconClipboard, IconClothing, IconDistance, IconFullBody, IconGaze, IconHeartPulse,
  IconInfo, IconLock, IconMail, IconPhone, IconPrinter, IconRefresh, IconRelaxed, IconRuler,
  IconShield, IconUpload, LogoMark, PictogramControl, PictogramMobility, PictogramStrength,
} from "./components/icons";
import { buildReport, smoothMetrics, type Metric, type ReportData, type ViewMode } from "./lib/posture";
import { getVideoLandmarker, getImageLandmarker, drawPose, avgVisibility, fallbackLandmarks, DEMO_IMAGES } from "./lib/poseEngine";
import { buildPrescription, findingLabelFor } from "./lib/exercises";
import { ANALYSIS_STAGES, DISCLAIMER, GRADE_COPY, RED_FLAGS, observationsFor } from "./lib/patientCopy";

type EngineState = "idle" | "loading" | "ready" | "error";
type StageMode = "idle" | "live" | "photo";
type Method = null | "camera" | "upload";

const HISTORY_KEY = "alignlab.history.v1";

function loadHistory(): HistoryEntry[] {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    return raw ? (JSON.parse(raw) as HistoryEntry[]) : [];
  } catch {
    return [];
  }
}

const GROUP_META: Record<string, { label: string; icon: React.ReactNode }> = {
  mobility: { label: "Mobility", icon: <PictogramMobility size={20} /> },
  activation: { label: "Motor control", icon: <PictogramControl size={20} /> },
  strength: { label: "Strength", icon: <PictogramStrength size={20} /> },
};

const VIEW_COPY: Record<ViewMode, { title: string; desc: string; note: string }> = {
  side: {
    title: "Side View",
    desc: "Best for assessing sagittal alignment patterns — head, shoulders, trunk, pelvis and knees from the side.",
    note: "Stand side-on to the camera, whichever side feels natural.",
  },
  front: {
    title: "Front View",
    desc: "Best for observing coronal alignment patterns — shoulder and hip level, trunk lean and knee tracking.",
    note: "Face the lens squarely with your weight even on both feet.",
  },
};

const POSITIONING_STEPS: { icon: React.ReactNode; text: string }[] = [
  { icon: <IconCameraHeight size={19} />, text: "Place the camera at approximately chest height." },
  { icon: <IconDistance size={19} />, text: "Stand around 2–3 metres from the camera." },
  { icon: <IconFullBody size={19} />, text: "Keep your whole body visible from head to feet." },
  { icon: <IconRelaxed size={19} />, text: "Stand naturally with your arms relaxed." },
  { icon: <IconGaze size={19} />, text: "Look straight ahead." },
  { icon: <IconClothing size={19} />, text: "Wear reasonably fitted clothing for clearer body alignment visibility." },
];

export default function App() {
  /* ---------------- state ---------------- */
  const [view, setView] = useState<ViewMode>("side");
  const [method, setMethod] = useState<Method>(null);
  const [engine, setEngine] = useState<EngineState>("idle");
  const [engineMsg, setEngineMsg] = useState<string | null>(null);
  const [stageMode, setStageMode] = useState<StageMode>("idle");
  const [processing, setProcessing] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [liveMetrics, setLiveMetrics] = useState<Metric[] | null>(null);
  const [confidence, setConfidence] = useState<number | null>(null);
  const [report, setReport] = useState<ReportData | null>(null);
  const [engineNote, setEngineNote] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>(loadHistory);
  const [flash, setFlash] = useState(0);
  const [stageAspect, setStageAspect] = useState("4 / 5");
  const [cameraFacing, setCameraFacing] = useState<"user" | "environment">("user");

  /* ---------------- refs ---------------- */
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const rafRef = useRef<number>(0);
  const lastLmsRef = useRef<NormalizedLandmark[] | null>(null);
  const smoothRef = useRef<Metric[] | null>(null);
  const lastUiRef = useRef(0);
  const viewRef = useRef(view);
  const loopOnRef = useRef(false);
  const fileRef = useRef<HTMLInputElement | null>(null);
  viewRef.current = view;

  const sessionId = useMemo(() => `YP-${Date.now().toString(36).slice(-5).toUpperCase()}`, []);

  /* ---------------- engine ---------------- */
  const ensureEngine = useCallback(async () => {
    if (engine === "ready" || engine === "loading") return;
    setEngine("loading");
    setEngineMsg(null);
    try {
      await getVideoLandmarker();
      setEngine("ready");
    } catch (e) {
      setEngine("error");
      setEngineMsg("The pose model could not be reached — camera scans are unavailable, but photo and demo scans will still work.");
      console.warn("Engine load failed", e);
    }
  }, [engine]);

  /* ---------------- live loop ---------------- */
  const loop = useCallback(async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;
    try {
      const lm = await getVideoLandmarker();
      loopOnRef.current = true;
      const step = () => {
        if (!loopOnRef.current) return;
        rafRef.current = requestAnimationFrame(step);
        if (video.readyState < 2 || video.videoWidth === 0) return;
        if (canvas.width !== video.videoWidth) {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
        }
        let res;
        try {
          res = lm.detectForVideo(video, performance.now());
        } catch {
          return;
        }
        const ctx = canvas.getContext("2d");
        if (!ctx) return;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const lms = res.landmarks?.[0];
        if (!lms || !lms.length) {
          setProcessing(false);
          smoothRef.current = null;
          return;
        }
        setProcessing(true);
        /* mirror to match the mirrored preview */
        const flipped = lms.map((p) => ({ ...p, x: 1 - p.x }));
        lastLmsRef.current = flipped;
        drawPose(ctx, flipped, canvas.width, canvas.height, { mirror: false });
        const now = performance.now();
        if (now - lastUiRef.current > 120) {
          lastUiRef.current = now;
          const raw = viewRef.current === "side" ? buildReport("side", flipped, 1).metrics : buildReport("front", flipped, 1).metrics;
          smoothRef.current = smoothMetrics(smoothRef.current, raw);
          setLiveMetrics(smoothRef.current);
          setConfidence(avgVisibility(lms));
        }
      };
      step();
    } catch (e) {
      setEngine("error");
      setEngineMsg("Could not start the live tracker on this device.");
      console.warn(e);
    }
  }, []);

  /* ---------------- camera ---------------- */
  const startCamera = useCallback(async () => {
    setError(null);
    ensureEngine();
    try {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
      }
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: cameraFacing }, width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      streamRef.current = stream;
      const video = videoRef.current;
      if (!video) return;
      video.srcObject = streamRef.current;
      await video.play();
      setStageAspect(`${video.videoWidth || 1280} / ${video.videoHeight || 720}`);
      setStageMode("live");
      setLiveMetrics(null);
      lastLmsRef.current = null;
      loop();
    } catch {
      setError("Camera unavailable or permission denied — you can upload a photo or try the demo scan instead.");
    }
  }, [ensureEngine, loop, cameraFacing]);

  const stopCamera = useCallback(() => {
    loopOnRef.current = false;
    cancelAnimationFrame(rafRef.current);
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    const video = videoRef.current;
    if (video) video.srcObject = null;
    const canvas = canvasRef.current;
    canvas?.getContext("2d")?.clearRect(0, 0, canvas.width, canvas.height);
    setStageMode("idle");
    setLiveMetrics(null);
    setProcessing(false);
    setConfidence(null);
    lastLmsRef.current = null;
    smoothRef.current = null;
  }, []);

  useEffect(() => () => stopCamera(), [stopCamera]);

  /* ---------------- report + history ---------------- */
  const finalizeReport = useCallback((r: ReportData) => {
    setReport(r);
    const flagged = r.metrics.filter((m) => m.severity > 0).sort((a, b) => b.severity - a.severity);
    const entry: HistoryEntry = {
      ts: r.ts,
      view: r.view,
      index: r.index,
      gradeLabel: r.grade.label,
      color: r.grade.color,
      top: flagged[0] ? findingLabelFor(flagged[0]) : "",
    };
    setHistory((h) => {
      const next = [...h, entry].slice(-30);
      try {
        localStorage.setItem(HISTORY_KEY, JSON.stringify(next));
      } catch { /* private mode */ }
      return next;
    });
    window.setTimeout(() => document.getElementById("results")?.scrollIntoView({ behavior: "smooth", block: "start" }), 250);
  }, []);

  /* ---------------- capture (freeze live frame) ---------------- */
  const capture = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const lms = lastLmsRef.current;
    if (!video || !canvas || !lms) {
      setError("We can't see your full body yet — step back until everything is in frame, then try again.");
      return;
    }
    loopOnRef.current = false;
    cancelAnimationFrame(rafRef.current);
    video.pause();
    setError(null);
    setAnalyzing(true);
    window.setTimeout(() => {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext("2d");
      if (ctx) {
        ctx.save();
        ctx.translate(canvas.width, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        ctx.restore();
        drawPose(ctx, lms, canvas.width, canvas.height, { mirror: false });
      }
      finalizeReport(buildReport(viewRef.current, lms, avgVisibility(lms)));
      setStageMode("photo");
      setProcessing(false);
      setLiveMetrics(null);
      setFlash((f) => f + 1);
      setAnalyzing(false);
    }, 1600);
  }, [finalizeReport]);

  /* ---------------- photo / demo analysis ---------------- */
  const recoverFromPhotoError = useCallback(
    (msg: string) => {
      setProcessing(false);
      setAnalyzing(false);
      setError(msg);
      if (streamRef.current && videoRef.current) {
        videoRef.current.play().catch(() => undefined);
        setStageMode("live");
        lastLmsRef.current = null;
        loop();
      } else {
        setStageMode("idle");
        const canvas = canvasRef.current;
        canvas?.getContext("2d")?.clearRect(0, 0, canvas.width, canvas.height);
      }
    },
    [loop]
  );

  const analyzeImage = useCallback(
    async (src: string, v: ViewMode, opts: { allowFallback: boolean; label: string }) => {
      setError(null);
      setEngineNote(null);
      setAnalyzing(true);
      /* pause live tracking so it does not repaint the canvas */
      loopOnRef.current = false;
      cancelAnimationFrame(rafRef.current);
      videoRef.current?.pause();
      setProcessing(true);
      setStageMode("photo");
      setLiveMetrics(null);
      setConfidence(null);
      smoothRef.current = null;

      const loadImg = (cors: boolean) =>
        new Promise<HTMLImageElement>((resolve, reject) => {
          const i = new Image();
          if (cors) i.crossOrigin = "anonymous";
          i.onload = () => resolve(i);
          i.onerror = () => reject(new Error("image load failed"));
          i.src = src;
        });

      let img: HTMLImageElement;
      let corsOk = true;
      try {
        img = await loadImg(true);
      } catch {
        try {
          img = await loadImg(false);
          corsOk = false;
        } catch {
          recoverFromPhotoError(`Could not load ${opts.label}. Please check your connection and try again.`);
          return;
        }
      }

      const canvas = canvasRef.current;
      if (!canvas) return;
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      setStageAspect(`${img.naturalWidth} / ${img.naturalHeight}`);
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.drawImage(img, 0, 0);

      let lms: NormalizedLandmark[] | null = null;
      let usedFallback = false;
      if (corsOk && engine !== "error") {
        try {
          ensureEngine();
          const lm = await getImageLandmarker();
          const res = lm.detect(img);
          lms = res.landmarks?.[0] ?? null;
        } catch {
          lms = null;
        }
      }
      if (!lms || !lms.length) {
        if (opts.allowFallback) {
          lms = fallbackLandmarks(v);
          usedFallback = true;
          setEngineNote(
            corsOk
              ? "No person was detected in this image, so an illustrative sample subject is shown to demonstrate the full screening flow."
              : "This image host blocked pixel access, so an illustrative sample subject is shown to demonstrate the full screening flow."
          );
        } else {
          recoverFromPhotoError("No person was detected in that image. Please use a photo where the whole body is visible, or try the demo scan.");
          return;
        }
      }
      drawPose(ctx, lms, canvas.width, canvas.height, { mirror: false });
      const conf = usedFallback ? 0.99 : avgVisibility(lms);
      finalizeReport(buildReport(v, lms, conf));
      setConfidence(conf);
      setFlash((f) => f + 1);
      await new Promise((r) => window.setTimeout(r, 1100));
      setProcessing(false);
      setAnalyzing(false);
    },
    [engine, ensureEngine, finalizeReport, recoverFromPhotoError]
  );

  const onUpload = useCallback(
    (file: File | undefined) => {
      if (!file) return;
      const url = URL.createObjectURL(file);
      analyzeImage(url, viewRef.current, { allowFallback: false, label: "that image" });
    },
    [analyzeImage]
  );

  const runDemo = useCallback(
    (v: ViewMode) => {
      setView(v);
      viewRef.current = v;
      setMethod(null);
      analyzeImage(DEMO_IMAGES[v], v, { allowFallback: true, label: "the demo subject" });
    },
    [analyzeImage]
  );

  const rx = useMemo(() => (report ? buildPrescription(report) : null), [report]);

  const reset = useCallback(() => {
    setReport(null);
    setEngineNote(null);
    setError(null);
    setMethod(null);
    stopCamera();
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, [stopCamera]);

  const scrollTo = (id: string) => document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });

  const chooseCamera = () => {
    setMethod("camera");
    setError(null);
    startCamera();
    window.setTimeout(() => scrollTo("step-3"), 60);
  };
  const chooseUpload = () => {
    setMethod("upload");
    setError(null);
    stopCamera();
    window.setTimeout(() => {
      scrollTo("step-3");
      fileRef.current?.click();
    }, 60);
  };

  const step = report || analyzing ? 4 : method ? 3 : 2;
  const tracked = processing && (confidence ?? 0) > 0.35;
  const statusLabel =
    stageMode === "live" ? (processing ? (tracked ? "Full body visible" : "Adjust slightly backward") : "Position yourself in frame")
    : stageMode === "photo" ? "Photo captured" : "Camera off";
  const statusColor = stageMode === "live" ? (processing ? (tracked ? "var(--color-scan)" : "var(--color-warn)") : "var(--color-info)") : stageMode === "photo" ? "var(--color-scan)" : "var(--color-faint)";

  const gradeCopy = report ? GRADE_COPY[report.grade.key] : null;
  let exNo = 0;

  /* ---------------- render ---------------- */
  return (
    <div className="clinic-bg min-h-screen">
      <AnalyzingOverlay active={analyzing} stages={ANALYSIS_STAGES} />
      <input ref={fileRef} type="file" accept="image/*" className="hidden" onChange={(e) => { onUpload(e.target.files?.[0]); e.target.value = ""; }} aria-label="Upload posture photo" />

      {/* Main Young Physio website provides the global header. */}

      <main id="top" className="mx-auto max-w-[1160px] px-4 pb-16 sm:px-6">
        {/* ============ hero / completion banner ============ */}
        {report && gradeCopy ? (
          <Reveal className="no-print">
            <div className="card mt-6 flex flex-wrap items-center gap-4 border-scan/25 bg-scan-soft/50 p-5 sm:p-6">
              <IconCheckCircle size={30} className="shrink-0 text-scan" />
              <div className="min-w-0 flex-1">
                <h2 className="font-display text-lg font-bold text-ink sm:text-xl">Screening complete — {gradeCopy.title}</h2>
                <p className="mt-0.5 text-[13.5px] text-muted">Your results and exercise plan are ready below. You can scan again any time.</p>
              </div>
              <div className="flex gap-2.5">
                <button onClick={() => scrollTo("results")} className="btn btn-primary !min-h-[44px] !px-5 !text-[14px]">Review results</button>
                <button onClick={reset} className="btn btn-outline !min-h-[44px] !px-5 !text-[14px]"><IconRefresh size={16} /> New screening</button>
              </div>
            </div>
          </Reveal>
        ) : (
          <section className="no-print grid items-center gap-10 pb-4 pt-10 sm:pt-14 lg:grid-cols-[1.05fr_0.95fr] lg:gap-14">
            <Reveal>
              <div className="eyebrow flex items-center gap-2">
                <IconHeartPulse size={15} /> FREE · TWO PHOTOS · ABOUT 2 MINUTES
              </div>
              <h1 className="mt-4 font-display text-[42px] font-bold leading-[1.02] tracking-tight text-ink sm:text-[64px]">
                See your posture<br />from head to <span className="text-accent">feet.</span>
              </h1>
              <p className="mt-6 max-w-[52ch] text-[16px] leading-relaxed text-muted">
                Upload one front and one side photo. Get a clear <strong>0–100 posture score</strong>, alignment insights, and physiotherapist-informed exercise guidance from Young Physio.
              </p>
              <div className="mt-6 flex flex-wrap gap-2">
                <span className="chip"><IconShield size={14} className="text-scan" /> Screening — not a diagnosis</span>
                <span className="chip"><IconLock size={14} className="text-accent-deep" /> Photos stay on your device</span>
                <span className="chip"><IconRuler size={14} className="text-info" /> Physio-informed reference ranges</span>
              </div>
              <div className="mt-8 flex flex-wrap items-center gap-3">
                <button onClick={() => scrollTo("step-1")} className="btn btn-accent">
                  Get My Posture Score <IconArrowRight size={17} />
                </button>
                <button onClick={() => runDemo(view)} className="btn btn-outline">Try a sample scan</button>
              </div>
              <p className="mt-4 text-[12.5px] font-medium text-faint">Takes about 2 minutes · No account needed</p>
            </Reveal>

            <Reveal delay={120}>
              <div className="hero-photo-card relative overflow-hidden rounded-[28px] border border-line-soft p-2 shadow-[var(--shadow-card)]">
                <div className="grid grid-cols-2 gap-2">
                  <div className="relative overflow-hidden rounded-[22px]">
                    <img src="https://res.cloudinary.com/jlb8fnn7/image/upload/v1788680091/ChatGPT_Image_Sep_6_2026_01_04_17_PM_omid1z.png" alt="Front posture analysis" className="hero-photo" />
                    <span className="hero-photo-label">FRONT VIEW</span>
                  </div>
                  <div className="relative overflow-hidden rounded-[22px]">
                    <img src="https://res.cloudinary.com/jlb8fnn7/image/upload/v1788680090/ChatGPT_Image_Sep_6_2026_01_04_06_PM_progoy.png" alt="Side posture analysis" className="hero-photo" />
                    <span className="hero-photo-label">SIDE VIEW</span>
                  </div>
                </div>
                <div className="hero-analysis-badge"><span className="h-2 w-2 rounded-full bg-scan" /> AI ANALYSIS</div>
              </div>
            </Reveal>
          </section>
        )}

        {/* ============ stepper ============ */}
        <div className="no-print sticky top-0 z-40 -mx-4 border-b border-line-soft bg-cream/92 px-4 py-3 backdrop-blur-md sm:-mx-6 sm:px-6">
          <Stepper current={step} />
        </div>

        {/* ============ step 1 · view ============ */}
        <section id="step-1" className="no-print scroll-mt-32 pt-10">
          <SectionHead kicker="Step 1" title="Choose Assessment View" sub="Pick the camera angle for this screening. You can run both views in separate sessions for a fuller picture." />
          <div className="grid gap-4 sm:grid-cols-2">
            {(["side", "front"] as ViewMode[]).map((v, i) => (
              <Reveal key={v} delay={i * 90}>
                <button
                  onClick={() => setView(v)}
                  aria-pressed={view === v}
                  className={`card card-hover group relative w-full overflow-hidden p-6 text-left transition-colors ${view === v ? "!border-accent bg-accent-soft/45" : ""}`}
                >
                  {view === v && (
                    <span className="absolute right-4 top-4 grid h-7 w-7 place-items-center rounded-full bg-accent text-white shadow-sm">
                      <IconCheck size={15} />
                    </span>
                  )}
                  <div className="flex items-center gap-6">
                    <div className={`grid h-[120px] w-[86px] shrink-0 place-items-center rounded-2xl border border-line-soft transition-colors ${view === v ? "bg-card" : "bg-well group-hover:bg-card"}`}>
                      {v === "side" ? <FigureSide className="h-[104px]" /> : <FigureFront className="h-[104px]" />}
                    </div>
                    <div>
                      <h3 className="font-display text-lg font-bold text-ink">{VIEW_COPY[v].title}</h3>
                      <p className="mt-1.5 text-[13.5px] leading-relaxed text-muted">{VIEW_COPY[v].desc}</p>
                    </div>
                  </div>
                </button>
              </Reveal>
            ))}
          </div>
        </section>

        {/* ============ step 2 · method ============ */}
        <section id="step-2" className="no-print scroll-mt-32 pt-12">
          <SectionHead kicker="Step 2" title="Capture Your Posture" sub="Use your camera for a guided live capture, or upload a clear full-body photo you already have." />
          <div className="grid gap-4 sm:grid-cols-2">
            <Reveal>
              <button onClick={chooseCamera} className={`card card-hover group w-full p-6 text-left sm:p-7 ${method === "camera" ? "!border-accent bg-accent-soft/45" : ""}`}>
                <div className="flex items-start gap-4">
                  <span className="grid h-12 w-12 shrink-0 place-items-center rounded-2xl bg-accent-soft text-accent-deep transition-transform group-hover:scale-105">
                    <IconCamera size={22} />
                  </span>
                  <div>
                    <h3 className="font-display text-lg font-bold text-ink">Use Camera</h3>
                    <p className="mt-1 text-[13.5px] leading-relaxed text-muted">Take a posture photo using your device camera, with live positioning guidance.</p>
                    <span className="mt-3 inline-flex items-center gap-1.5 text-[13px] font-bold text-accent-deep">
                      Start camera <IconArrowRight size={15} className="transition-transform group-hover:translate-x-0.5" />
                    </span>
                  </div>
                </div>
              </button>
            </Reveal>
            <Reveal delay={90}>
              <button onClick={chooseUpload} className={`card card-hover group w-full p-6 text-left sm:p-7 ${method === "upload" ? "!border-accent bg-accent-soft/45" : ""}`}>
                <div className="flex items-start gap-4">
                  <span className="grid h-12 w-12 shrink-0 place-items-center rounded-2xl bg-scan-soft text-scan transition-transform group-hover:scale-105">
                    <IconUpload size={22} />
                  </span>
                  <div>
                    <h3 className="font-display text-lg font-bold text-ink">Upload Photo</h3>
                    <p className="mt-1 text-[13.5px] leading-relaxed text-muted">Upload a clear full-body posture image — JPG or PNG, taken from the side or front.</p>
                    <span className="mt-3 inline-flex items-center gap-1.5 text-[13px] font-bold text-accent-deep">
                      Choose a photo <IconArrowRight size={15} className="transition-transform group-hover:translate-x-0.5" />
                    </span>
                  </div>
                </div>
              </button>
            </Reveal>
          </div>
          <p className="mt-4 text-[13.5px] text-muted">
            Not ready to use your camera?{" "}
            <button onClick={() => runDemo(view)} className="font-bold text-accent-deep underline decoration-accent/50 underline-offset-4 transition-colors hover:text-accent">
              Explore the full flow with a demo subject
            </button>
          </p>
        </section>

        {/* ============ step 3 · positioning + capture ============ */}
        {method && (
          <section id="step-3" className="no-print scroll-mt-32 pt-12">
            <SectionHead
              kicker="Step 3"
              title="Position Yourself Correctly"
              sub="Good positioning is the key to a reliable screening. A quick checklist before you capture:"
            />
            <div className="grid gap-5 lg:grid-cols-[400px_minmax(0,1fr)]">
              {/* instructions */}
              <Reveal>
                <div className="h-full rounded-[22px] border border-line-soft bg-sand p-6">
                  <h3 className="font-display text-[17px] font-bold text-ink">How to Position Yourself</h3>
                  <ol className="mt-4 space-y-3.5">
                    {POSITIONING_STEPS.map((s, i) => (
                      <li key={i} className="flex items-start gap-3">
                        <span className="grid h-9 w-9 shrink-0 place-items-center rounded-xl border border-line-soft bg-card text-accent-deep shadow-sm">
                          {s.icon}
                        </span>
                        <span className="pt-1.5 text-[13.5px] font-medium leading-relaxed text-ink-soft">{s.text}</span>
                      </li>
                    ))}
                  </ol>
                  <p className="mt-5 flex items-start gap-2 rounded-xl border border-accent/25 bg-accent-soft/70 p-3 text-[12.5px] font-semibold leading-relaxed text-accent-deep">
                    <IconInfo size={15} className="mt-0.5 shrink-0" />
                    {VIEW_COPY[view].note} You're screening the {view} view — switch in Step 1 if needed.
                  </p>
                </div>
              </Reveal>

              {/* stage + controls */}
              <Reveal delay={100}>
                <div key={flash} className="tick-in space-y-4">
                  <Stage
                    videoRef={videoRef}
                    canvasRef={canvasRef}
                    mode={stageMode}
                    processing={processing}
                    mirrored
                    aspect={stageAspect}
                    statusLabel={statusLabel}
                    statusColor={statusColor}
                    confidence={confidence}
                    view={view}
                  />

                  {error && (
                    <div className="flex items-start gap-3 rounded-2xl border border-risk/30 bg-risk-soft/70 p-4">
                      <IconAlert size={18} className="mt-0.5 shrink-0 text-risk" />
                      <p className="text-[13.5px] font-semibold leading-relaxed text-ink-soft">{error}</p>
                    </div>
                  )}
                  {engineNote && (
                    <div className="flex items-start gap-3 rounded-2xl border border-info/30 bg-info-soft/70 p-4">
                      <IconInfo size={18} className="mt-0.5 shrink-0 text-info" />
                      <p className="text-[13.5px] font-semibold leading-relaxed text-ink-soft">{engineNote}</p>
                    </div>
                  )}
                  {engineMsg && (
                    <div className="flex items-start gap-3 rounded-2xl border border-warn/30 bg-warn-soft/70 p-4">
                      <IconAlert size={18} className="mt-0.5 shrink-0 text-warn" />
                      <p className="text-[13.5px] font-semibold leading-relaxed text-ink-soft">{engineMsg}</p>
                    </div>
                  )}

                  <div className="card flex flex-wrap items-center gap-3 p-4">
                    {method === "camera" ? (
                      <>
                        <button onClick={capture} disabled={!processing || analyzing} className="btn btn-accent flex-1 !text-[15.5px] sm:flex-none sm:!px-10">
                          <IconCamera size={18} /> Capture Photo
                        </button>
                        <button onClick={() => {
                          const next = cameraFacing === "user" ? "environment" : "user";
                          setCameraFacing(next);
                          stopCamera();
                          window.setTimeout(() => startCamera(), 100);
                        }} className="btn btn-soft">
                          {cameraFacing === "user" ? "Use Back Camera" : "Use Front Camera"}
                        </button>
                        <button onClick={() => { stopCamera(); }} className="btn btn-soft">Stop camera</button>
                        <button onClick={() => { setMethod(null); stopCamera(); }} className="btn btn-outline !px-4">Change method</button>
                        {stageMode === "live" && liveMetrics && (
                          <span className="ml-auto text-[12px] font-semibold text-faint">
                            Tracking {liveMetrics.length} alignment measures in real time
                          </span>
                        )}
                      </>
                    ) : (
                      <>
                        <button onClick={() => fileRef.current?.click()} className="btn btn-accent flex-1 !text-[15.5px] sm:flex-none sm:!px-10">
                          <IconUpload size={18} /> {stageMode === "photo" ? "Choose another photo" : "Choose photo"}
                        </button>
                        <button onClick={() => setMethod(null)} className="btn btn-outline !px-4">Change method</button>
                        <span className="ml-auto text-[12px] font-semibold text-faint">JPG or PNG · whole body visible</span>
                      </>
                    )}
                  </div>
                </div>
              </Reveal>
            </div>
          </section>
        )}

        {/* ============ step 4 · results ============ */}
        {report && rx && gradeCopy && (
          <section id="results" className="print-area scroll-mt-24 pt-14">
            <SectionHead
              kicker="Step 4 · Your results"
              title="Your Posture Screening"
              sub="A structured view of the alignment patterns observed in your image — and what you can do about them."
            />

            {/* overview */}
            <Reveal>
              <div className="card overflow-hidden">
                <div className="grid gap-6 p-6 sm:p-8 md:grid-cols-[auto_1fr] md:items-center">
                  <div className="mx-auto">
                    <IndexRing index={report.index} color={report.grade.color} />
                  </div>
                  <div>
                    <div className="eyebrow">Posture Overview</div>
                    <div className="mt-2 flex flex-wrap items-center gap-3">
                      <h3 className="font-display text-[26px] font-bold tracking-tight text-ink sm:text-3xl">{gradeCopy.title}</h3>
                      <span
                        className="rounded-full border px-3 py-1 text-[12.5px] font-bold"
                        style={{ color: report.grade.color, borderColor: `color-mix(in srgb, ${report.grade.color} 35%, transparent)`, background: `color-mix(in srgb, ${report.grade.color} 9%, transparent)` }}
                      >
                        Screening observation
                      </span>
                    </div>
                    <p className="mt-3 max-w-[62ch] text-[15px] leading-relaxed text-muted">{gradeCopy.blurb}</p>
                    <div className="mt-5 flex flex-wrap gap-2">
                      <span className="chip">{report.view === "side" ? "Side view · sagittal plane" : "Front view · coronal plane"}</span>
                      <span className="chip">{report.metrics.length} alignment measures</span>
                      <span className="chip">Image clarity <span className="value-mono">{Math.round(report.confidence * 100)}%</span></span>
                      <span className="chip">{new Date(report.ts).toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" })}</span>
                    </div>
                    <div className="no-print mt-6 flex flex-wrap gap-3">
                      <button onClick={() => window.print()} className="btn btn-primary"><IconPrinter size={17} /> Print report</button>
                      <button onClick={reset} className="btn btn-outline"><IconRefresh size={16} /> Start a new screening</button>
                    </div>
                  </div>
                </div>
              </div>
            </Reveal>

            {/* key observations */}
            <div className="pt-12">
              <SectionHead kicker="Screening observations" title="Key Observations" sub="A quick, plain-language summary of what was visible in this image." />
              <ObservationList observations={observationsFor(report)} />
            </div>

            {/* attention */}
            <div className="pt-12">
              <SectionHead kicker="Prioritised for you" title="What Needs Attention" sub="Findings ranked by prominence. 'Variation' simply means the pattern differs from typical reference ranges." />
              <AttentionList metrics={report.metrics} />
            </div>

            {/* detailed measurements */}
            <div className="pt-12">
              <SectionHead
                kicker="The detail"
                title="Detailed Measurements"
                sub="Each measure compares what's visible in your image with published photographic reference ranges. Open 'Clinical information' for the scientific context."
              />
              <div className="grid gap-4 sm:grid-cols-2">
                {report.metrics.map((m, i) => (
                  <MetricCard key={m.id} m={m} i={i} />
                ))}
              </div>
            </div>

            {/* exercise plan */}
            <div id="plan" className="pt-14">
              <SectionHead
                kicker="Your plan"
                title="Your Recommended Exercise Plan"
                sub={`These exercises are general movement recommendations based on the visible screening patterns — roughly ${rx.totalMinutes} focused minutes per session. Stop if you feel significant pain.`}
              />
              <div className="space-y-8">
                {rx.groups.map((g) => (
                  <div key={g.key}>
                    <Reveal>
                      <div className="mb-4 flex items-start gap-3">
                        <span className="grid h-10 w-10 shrink-0 place-items-center rounded-xl border border-line-soft bg-card text-accent-deep shadow-sm">
                          {GROUP_META[g.key]?.icon}
                        </span>
                        <div>
                          <h4 className="font-display text-[17px] font-bold text-ink">{GROUP_META[g.key]?.label ?? g.title}</h4>
                          <p className="mt-0.5 max-w-[70ch] text-[12.5px] font-medium leading-relaxed text-faint">{g.note}</p>
                        </div>
                      </div>
                    </Reveal>
                    <div className="grid gap-4 lg:grid-cols-2">
                      {g.items.map((item) => {
                        exNo += 1;
                        return <ExerciseCard key={item.ex.id} item={item} n={exNo} />;
                      })}
                    </div>
                  </div>
                ))}
              </div>

              <div className="pt-10">
                <Reveal>
                  <h4 className="mb-4 font-display text-[17px] font-bold text-ink">Suggested weekly rhythm</h4>
                </Reveal>
                <ScheduleStrip rx={rx} />
                <Reveal delay={120}>
                  <div className="card mt-4 p-5 sm:p-6">
                    <div className="eyebrow !text-[10px]">Guidelines for your plan</div>
                    <ul className="mt-3 grid gap-x-8 gap-y-2.5 sm:grid-cols-2">
                      {rx.rules.map((r, i) => (
                        <li key={i} className="flex items-start gap-2.5 text-[13.5px] leading-relaxed text-muted">
                          <IconCheck size={15} className="mt-0.5 shrink-0 text-scan" /> {r}
                        </li>
                      ))}
                    </ul>
                  </div>
                </Reveal>
              </div>
            </div>

            {/* safety + disclaimer */}
            <div className="pt-14">
              <div className="grid gap-4 lg:grid-cols-2">
                <Reveal>
                  <div className="card h-full p-6 sm:p-7">
                    <div className="flex items-center gap-2.5">
                      <span className="grid h-10 w-10 place-items-center rounded-xl bg-risk-soft text-risk"><IconAlert size={19} /></span>
                      <h4 className="font-display text-[17px] font-bold text-ink">When to seek professional care</h4>
                    </div>
                    <ul className="mt-4 space-y-2.5">
                      {RED_FLAGS.map((f, i) => (
                        <li key={i} className="flex gap-2.5 text-[13.5px] leading-relaxed text-muted">
                          <span className="mt-[7px] h-1.5 w-1.5 shrink-0 rounded-full bg-risk/70" /> {f}
                        </li>
                      ))}
                    </ul>
                    <p className="mt-4 border-t border-line-soft pt-3.5 text-[12.5px] leading-relaxed text-faint">
                      Any of these signs warrants prompt evaluation by a doctor or physiotherapist — please don't exercise through them.
                    </p>
                  </div>
                </Reveal>
                <Reveal delay={90}>
                  <div className="card h-full p-6 sm:p-7">
                    <div className="flex items-center gap-2.5">
                      <span className="grid h-10 w-10 place-items-center rounded-xl bg-scan-soft text-scan"><IconShield size={19} /></span>
                      <h4 className="font-display text-[17px] font-bold text-ink">About this screening</h4>
                    </div>
                    <p className="mt-4 text-[13.5px] leading-relaxed text-muted">
                      This is a photographic posture screening: an on-device AI maps 33 body points from your image and compares visible
                      alignment patterns with ranges reported in the physiotherapy literature. It observes patterns — it cannot diagnose
                      conditions, and measurements like spinal curvature require clinical examination. Bring this report to a qualified
                      physiotherapist for personalised advice.
                    </p>
                    <p className="mt-3.5 flex items-center gap-2 rounded-xl bg-sand/80 p-3 text-[12.5px] font-semibold text-ink-soft">
                      <IconLock size={15} className="shrink-0 text-accent-deep" /> Your photo is processed in your browser and never uploaded or stored.
                    </p>
                  </div>
                </Reveal>
              </div>
              <Reveal delay={120}>
                <div className="mt-4 rounded-[22px] border border-accent/25 bg-accent-soft/60 p-6">
                  <div className="flex items-start gap-3">
                    <IconInfo size={20} className="mt-0.5 shrink-0 text-accent-deep" />
                    <div>
                      <h4 className="font-display text-[15px] font-bold text-ink">Important information</h4>
                      <p className="mt-1.5 text-[13.5px] leading-relaxed text-ink-soft">{DISCLAIMER}</p>
                    </div>
                  </div>
                </div>
              </Reveal>
            </div>

            {/* CTA */}
            <Reveal className="no-print">
              <div className="mt-14 overflow-hidden rounded-[26px] bg-ink shadow-[var(--shadow-lift)]">
                <div className="relative grid items-center gap-6 p-8 sm:p-10 md:grid-cols-[1fr_auto]">
                  <div className="dot-field pointer-events-none absolute inset-0 opacity-40" aria-hidden />
                  <div className="relative">
                    <div className="eyebrow !text-accent">Young Physio</div>
                    <h3 className="mt-2 font-display text-[26px] font-bold tracking-tight text-cream sm:text-3xl">Need Professional Guidance?</h3>
                    <p className="mt-2.5 max-w-[52ch] text-[15px] leading-relaxed text-cream/70">
                      Get personalised advice from a qualified physiotherapist — they'll interpret these findings alongside your symptoms,
                      history and movement testing.
                    </p>
                  </div>
                  <div className="relative flex flex-wrap gap-3">
                    <a href="mailto:care@youngphysio.example?subject=Posture%20screening%20consultation" className="btn btn-accent">
                      <IconMail size={17} /> Book a Consultation
                    </a>
                    <a href="tel:+10000000000" className="btn border border-cream/25 bg-transparent text-cream hover:bg-cream/10">
                      <IconPhone size={17} /> Talk to a Physiotherapist
                    </a>
                  </div>
                </div>
              </div>
            </Reveal>

            {/* history */}
            <div id="log" className="no-print pt-14">
              <HistoryPanel
                entries={history}
                onClear={() => {
                  setHistory([]);
                  try { localStorage.removeItem(HISTORY_KEY); } catch { /* ignore */ }
                }}
              />
            </div>
          </section>
        )}

        {/* ============ pre-results placeholders ============ */}
        {!report && (
          <section className="no-print pt-14">
            <Reveal>
              <div className="grid gap-4 sm:grid-cols-3">
                {[
                  { icon: <IconClipboard size={20} />, t: "Structured results", d: "Every screening produces a clear overview, ranked observations and detailed measures." },
                  { icon: <IconHeartPulse size={20} />, t: "A plan you can follow", d: "Mobility, motor-control and strength exercises dosed in sets, reps and holds." },
                  { icon: <IconShield size={20} />, t: "Private by design", d: "Analysis runs on your device. Nothing is uploaded, stored or shared." },
                ].map((f, i) => (
                  <div key={i} className="card card-hover p-5">
                    <span className="grid h-11 w-11 place-items-center rounded-xl bg-accent-soft text-accent-deep">{f.icon}</span>
                    <h4 className="mt-3.5 font-display text-[15.5px] font-bold text-ink">{f.t}</h4>
                    <p className="mt-1.5 text-[13.5px] leading-relaxed text-muted">{f.d}</p>
                  </div>
                ))}
              </div>
            </Reveal>
          </section>
        )}
      </main>

      {/* ============ footer ============ */}
      <footer className="border-t border-line-soft bg-sand/60">
        <div className="mx-auto max-w-[1160px] px-4 py-10 sm:px-6">
          <div className="flex flex-wrap items-center justify-between gap-6">
            <div className="flex items-center gap-3">
              <LogoMark size={34} />
              <div>
                <div className="font-display text-[15px] font-bold text-ink">Young Physio — AI Posture Analysis</div>
                <div className="mt-0.5 text-[12px] font-medium text-faint">Educational posture screening · Not a medical diagnosis · Session {sessionId}</div>
              </div>
            </div>
            <div className="flex items-center gap-2 text-[12px] font-semibold text-faint">
              <IconLock size={14} className="text-accent-deep" /> On-device analysis · No data leaves your browser
            </div>
          </div>
          <p className="mt-6 max-w-[90ch] text-[11.5px] leading-relaxed text-faint">
            {DISCLAIMER}
          </p>
          <p className="mt-3 text-[11.5px] text-faint">© {new Date().getFullYear()} Young Physio. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
}
