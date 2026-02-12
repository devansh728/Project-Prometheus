import { useRef, useState, useCallback } from 'react';
import { Platform } from 'react-native';

// The HTML that runs inside the WebView
export const SPEECH_RECOGNITION_HTML = `
<!DOCTYPE html>
<html>
<head>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    body {
      margin: 0;
      padding: 0;
      background: transparent;
      overflow: hidden;
    }
  </style>
</head>
<body>
<script>
  let recognition = null;
  let isListening = false;

  function initRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    
    if (!SpeechRecognition) {
      window.ReactNativeWebView.postMessage(JSON.stringify({
        type: 'error',
        message: 'SpeechRecognition not supported in this WebView'
      }));
      return null;
    }

    const rec = new SpeechRecognition();
    rec.continuous = true;
    rec.interimResults = true;
    rec.lang = 'en-US';
    rec.maxAlternatives = 3;

    rec.onstart = function() {
      isListening = true;
      window.ReactNativeWebView.postMessage(JSON.stringify({
        type: 'status',
        status: 'listening'
      }));
    };

    rec.onresult = function(event) {
      let interimTranscript = '';
      let finalTranscript = '';

      for (let i = event.resultIndex; i < event.results.length; i++) {
        const transcript = event.results[i][0].transcript;
        const confidence = event.results[i][0].confidence;
        
        if (event.results[i].isFinal) {
          finalTranscript += transcript;
          window.ReactNativeWebView.postMessage(JSON.stringify({
            type: 'final',
            text: transcript.trim(),
            confidence: confidence,
            allAlternatives: Array.from(event.results[i]).map(alt => ({
              text: alt.transcript.trim(),
              confidence: alt.confidence
            }))
          }));
        } else {
          interimTranscript += transcript;
          window.ReactNativeWebView.postMessage(JSON.stringify({
            type: 'interim',
            text: interimTranscript.trim()
          }));
        }
      }
    };

    rec.onerror = function(event) {
      window.ReactNativeWebView.postMessage(JSON.stringify({
        type: 'error',
        message: event.error
      }));
      
      // Auto-restart on non-fatal errors
      if (event.error === 'no-speech' && isListening) {
        setTimeout(() => {
          try { rec.start(); } catch(e) {}
        }, 100);
      }
    };

    rec.onend = function() {
      // Auto-restart if still supposed to be listening
      if (isListening) {
        try {
          setTimeout(() => rec.start(), 100);
        } catch(e) {}
      } else {
        window.ReactNativeWebView.postMessage(JSON.stringify({
          type: 'status',
          status: 'stopped'
        }));
      }
    };

    return rec;
  }

  // Listen for commands from React Native
  document.addEventListener('message', function(event) {
    const data = JSON.parse(event.data);
    
    if (data.command === 'start') {
      if (!recognition) {
        recognition = initRecognition();
      }
      if (recognition) {
        isListening = true;
        try {
          recognition.start();
        } catch(e) {
          // Already started, ignore
        }
      }
    } else if (data.command === 'stop') {
      isListening = false;
      if (recognition) {
        try {
          recognition.stop();
        } catch(e) {}
      }
    }
  });

  // Also handle window.onmessage for Android
  window.addEventListener('message', function(event) {
    try {
      const data = JSON.parse(event.data);
      if (data.command === 'start') {
        if (!recognition) {
          recognition = initRecognition();
        }
        if (recognition) {
          isListening = true;
          try { recognition.start(); } catch(e) {}
        }
      } else if (data.command === 'stop') {
        isListening = false;
        if (recognition) {
          try { recognition.stop(); } catch(e) {}
        }
      }
    } catch(e) {}
  });

  // Signal ready
  window.ReactNativeWebView.postMessage(JSON.stringify({
    type: 'status',
    status: 'ready'
  }));
</script>
</body>
</html>
`;

// Fuzzy matching utility
export function fuzzyMatch(
  spokenText: string,
  options: string[],
  threshold: number = 0.35
): { match: string | null; score: number; index: number } {
  
  const normalize = (s: string) =>
    s.toLowerCase()
      .replace(/[^a-z0-9\s]/g, '')
      .replace(/\s+/g, ' ')
      .trim();

  const spoken = normalize(spokenText);
  const spokenWords = spoken.split(' ');

  let bestMatch: string | null = null;
  let bestScore = 0;
  let bestIndex = -1;

  for (let i = 0; i < options.length; i++) {
    const option = normalize(options[i]);
    const optionWords = option.split(' ');

    // Strategy 1: Direct substring containment
    if (spoken.includes(option) || option.includes(spoken)) {
      const score = 0.95;
      if (score > bestScore) {
        bestScore = score;
        bestMatch = options[i];
        bestIndex = i;
      }
      continue;
    }

    // Strategy 2: Word overlap (Jaccard-like)
    const spokenSet = new Set(spokenWords);
    const optionSet = new Set(optionWords);
    let overlap = 0;
    for (const word of optionSet) {
      if (spokenSet.has(word)) overlap++;
    }
    const jaccard = overlap / Math.max(optionSet.size, 1);

    // Strategy 3: Key phrase detection
    const keyPhrases: Record<string, string[]> = {
      'book': ['book', 'schedule', 'appointment', 'yes please', 'go ahead', 'do it'],
      'cost': ['cost', 'price', 'how much', 'expensive', 'charge', 'money', 'pay'],
      'safe': ['safe', 'drive', 'okay to drive', 'danger', 'risk'],
      'time': ['time', 'when', 'available', 'slot', 'tomorrow', 'today'],
      'serious': ['serious', 'bad', 'severe', 'worry', 'concerned'],
      'wrong': ['wrong', 'issue', 'problem', 'what happened', 'whats'],
      'recommend': ['recommend', 'suggest', 'should i', 'what do', 'advice'],
      'busy': ['busy', 'later', 'not now', 'call back', 'cant talk'],
      'send': ['send', 'message', 'text', 'details', 'email', 'notification'],
      'thanks': ['thanks', 'thank you', 'great', 'perfect', 'good', 'bye', 'done'],
      'expect': ['expect', 'what happens', 'process', 'procedure'],
      'wait': ['wait', 'postpone', 'delay', 'next week', 'not yet'],
    };

    let phraseScore = 0;
    for (const [, phrases] of Object.entries(keyPhrases)) {
      const optionHasPhrase = phrases.some(p => option.includes(p));
      const spokenHasPhrase = phrases.some(p => spoken.includes(p));
      if (optionHasPhrase && spokenHasPhrase) {
        phraseScore = 0.85;
        break;
      }
    }

    const combinedScore = Math.max(jaccard, phraseScore);

    if (combinedScore > bestScore) {
      bestScore = combinedScore;
      bestMatch = options[i];
      bestIndex = i;
    }
  }

  if (bestScore >= threshold) {
    return { match: bestMatch, score: bestScore, index: bestIndex };
  }

  return { match: null, score: bestScore, index: -1 };
}