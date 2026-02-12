/**
 * Voice Agent Service - TTS with Emotional Tone Mapping
 * Uses Web Speech API with fallback to pre-recorded audio
 */

export type EmotionalTone = 'calm' | 'concerned' | 'urgent' | 'friendly' | 'empathetic';

interface VoiceConfig {
  rate: number;
  pitch: number;
  volume: number;
}

const TONE_CONFIGS: Record<EmotionalTone, VoiceConfig> = {
  calm: { rate: 0.9, pitch: 1.0, volume: 0.8 },
  concerned: { rate: 0.95, pitch: 1.1, volume: 0.85 },
  urgent: { rate: 1.1, pitch: 1.2, volume: 1.0 },
  friendly: { rate: 1.0, pitch: 1.05, volume: 0.9 },
  empathetic: { rate: 0.85, pitch: 0.95, volume: 0.75 },
};

// Pre-recorded fallback scripts for demo reliability
const FALLBACK_SCRIPTS: Record<string, { text: string; tone: EmotionalTone }[]> = {
  critical_alert: [
    { text: "Hello, this is SentinEV calling regarding your vehicle.", tone: 'friendly' },
    { text: "We've detected a potential brake system issue that requires your attention.", tone: 'concerned' },
    { text: "For your safety, we recommend scheduling a service appointment soon.", tone: 'empathetic' },
    { text: "Would you like me to book the earliest available slot for you?", tone: 'friendly' },
  ],
  appointment_reminder: [
    { text: "Hi, this is a friendly reminder from SentinEV.", tone: 'friendly' },
    { text: "Your scheduled service appointment is tomorrow at 10 AM.", tone: 'calm' },
    { text: "Please arrive 15 minutes early for check-in.", tone: 'calm' },
  ],
  service_complete: [
    { text: "Great news! Your vehicle service has been completed.", tone: 'friendly' },
    { text: "All brake components have been replaced and tested.", tone: 'calm' },
    { text: "Your vehicle is ready for pickup at your convenience.", tone: 'friendly' },
  ],
  emergency: [
    { text: "This is an urgent safety alert from SentinEV.", tone: 'urgent' },
    { text: "We've detected a critical battery thermal event in your vehicle.", tone: 'urgent' },
    { text: "Please pull over safely and contact emergency services if needed.", tone: 'empathetic' },
    { text: "A service team has been dispatched to your location.", tone: 'calm' },
  ],
};

class VoiceAgentService {
  private synth: SpeechSynthesis | null = null;
  private voices: SpeechSynthesisVoice[] = [];
  private preferredVoice: SpeechSynthesisVoice | null = null;
  private isInitialized = false;
  private isSpeaking = false;
  private speechQueue: { text: string; tone: EmotionalTone }[] = [];

  constructor() {
    if (typeof window !== 'undefined' && window.speechSynthesis) {
      this.synth = window.speechSynthesis;
      this.loadVoices();
    }
  }

  private loadVoices(): void {
    if (!this.synth) return;

    const loadVoicesInternal = () => {
      this.voices = this.synth!.getVoices();
      // Prefer high-quality voices
      this.preferredVoice = this.voices.find(v => 
        v.name.includes('Google') && v.lang.startsWith('en')
      ) || this.voices.find(v => 
        v.lang.startsWith('en') && v.localService === false
      ) || this.voices.find(v => 
        v.lang.startsWith('en')
      ) || null;
      
      this.isInitialized = this.voices.length > 0;
    };

    loadVoicesInternal();
    if (this.voices.length === 0) {
      this.synth.addEventListener('voiceschanged', loadVoicesInternal);
    }
  }

  /**
   * Speak text with emotional tone
   */
  async speak(text: string, tone: EmotionalTone = 'calm'): Promise<void> {
    return new Promise((resolve, reject) => {
      if (!this.synth || !this.isInitialized) {
        console.warn('Speech synthesis not available');
        resolve();
        return;
      }

      const config = TONE_CONFIGS[tone];
      const utterance = new SpeechSynthesisUtterance(text);
      
      if (this.preferredVoice) {
        utterance.voice = this.preferredVoice;
      }
      
      utterance.rate = config.rate;
      utterance.pitch = config.pitch;
      utterance.volume = config.volume;

      utterance.onend = () => {
        this.isSpeaking = false;
        resolve();
      };

      utterance.onerror = (event) => {
        this.isSpeaking = false;
        reject(event);
      };

      this.isSpeaking = true;
      this.synth.speak(utterance);
    });
  }

  /**
   * Play a predefined script with proper pacing
   */
  async playScript(scriptKey: keyof typeof FALLBACK_SCRIPTS): Promise<void> {
    const script = FALLBACK_SCRIPTS[scriptKey];
    if (!script) {
      console.error(`Script "${scriptKey}" not found`);
      return;
    }

    for (const line of script) {
      await this.speak(line.text, line.tone);
      await this.delay(500); // Pause between lines
    }
  }

  /**
   * Generate dynamic response based on context
   */
  async speakDynamic(context: {
    customerName?: string;
    vehicleModel?: string;
    issueType?: string;
    severity?: 'info' | 'warning' | 'critical';
    appointmentTime?: string;
  }): Promise<void> {
    const { customerName = 'valued customer', vehicleModel, issueType, severity = 'info', appointmentTime } = context;
    
    const tone: EmotionalTone = severity === 'critical' ? 'urgent' : severity === 'warning' ? 'concerned' : 'friendly';
    
    await this.speak(`Hello ${customerName}, this is SentinEV's intelligent vehicle assistant.`, 'friendly');
    await this.delay(400);
    
    if (vehicleModel && issueType) {
      const issueMessage = severity === 'critical'
        ? `We've detected a critical ${issueType} issue in your ${vehicleModel} that requires immediate attention.`
        : `We've noticed some ${issueType} degradation in your ${vehicleModel} that you should be aware of.`;
      await this.speak(issueMessage, tone);
      await this.delay(400);
    }
    
    if (appointmentTime) {
      await this.speak(`I've reserved an appointment slot for ${appointmentTime}.`, 'calm');
      await this.speak('Would you like me to confirm this booking?', 'friendly');
    } else {
      await this.speak('Would you like me to schedule a service appointment for you?', 'friendly');
    }
  }

  /**
   * Stop all speech
   */
  stop(): void {
    if (this.synth) {
      this.synth.cancel();
      this.isSpeaking = false;
    }
  }

  /**
   * Check if currently speaking
   */
  get speaking(): boolean {
    return this.isSpeaking;
  }

  /**
   * Check if TTS is available
   */
  get available(): boolean {
    return this.isInitialized;
  }

  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// Singleton instance
export const voiceAgent = new VoiceAgentService();

// Export types and fallback scripts for testing
export { FALLBACK_SCRIPTS, TONE_CONFIGS };
