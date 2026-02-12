import React, { useRef, forwardRef, useImperativeHandle } from 'react';
import { StyleSheet, Platform } from 'react-native';
import { WebView } from 'react-native-webview';
import { SPEECH_RECOGNITION_HTML } from '../../hooks/useWebViewSpeechRecognition';

export interface SpeechWebViewRef {
  startListening: () => void;
  stopListening: () => void;
}

interface SpeechWebViewProps {
  onInterimResult: (text: string) => void;
  onFinalResult: (text: string, confidence: number) => void;
  onStatusChange: (status: string) => void;
  onError: (error: string) => void;
}

export const SpeechWebView = forwardRef<SpeechWebViewRef, SpeechWebViewProps>(
  ({ onInterimResult, onFinalResult, onStatusChange, onError }, ref) => {
    const webViewRef = useRef<WebView>(null);

    useImperativeHandle(ref, () => ({
      startListening: () => {
        webViewRef.current?.postMessage(
          JSON.stringify({ command: 'start' })
        );
      },
      stopListening: () => {
        webViewRef.current?.postMessage(
          JSON.stringify({ command: 'stop' })
        );
      },
    }));

    const handleMessage = (event: any) => {
      try {
        const data = JSON.parse(event.nativeEvent.data);

        switch (data.type) {
          case 'interim':
            onInterimResult(data.text);
            break;
          case 'final':
            onFinalResult(data.text, data.confidence);
            break;
          case 'status':
            onStatusChange(data.status);
            break;
          case 'error':
            onError(data.message);
            break;
        }
      } catch (e) {
        console.error('WebView message parse error:', e);
      }
    };

    return (
      <WebView
        ref={webViewRef}
        source={{ html: SPEECH_RECOGNITION_HTML }}
        style={styles.hidden}
        onMessage={handleMessage}
        javaScriptEnabled={true}
        mediaPlaybackRequiresUserAction={false}
        allowsInlineMediaPlayback={true}
        // Android: allow microphone access
        allowFileAccess={true}
        // These properties help with mic permissions on Android WebView
        originWhitelist={['*']}
        // @ts-ignore
        onPermissionRequest={(event: any) => {
          // Auto-grant microphone permission
          event?.grant?.();
        }}
      />
    );
  }
);

const styles = StyleSheet.create({
  hidden: {
    position: 'absolute',
    width: 0,
    height: 0,
    opacity: 0,
    // Place it off-screen but keep it alive
    top: -1000,
    left: -1000,
  },
});