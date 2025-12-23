
import React from 'react';
import { AbsoluteFill, Audio, Img, Sequence, useCurrentFrame, useVideoConfig } from 'remotion';

// ===================================
// Helper Components
// ===================================

const Subtitle: React.FC<{ text: string }> = ({ text }) => {
    return (
        <div style={{
            position: 'absolute',
            bottom: 100,
            width: '100%',
            textAlign: 'center',
            fontSize: 60,
            color: 'white',
            textShadow: '2px 2px 4px rgba(0,0,0,0.8)',
            fontFamily: 'Helvetica, Arial, sans-serif',
            fontWeight: 'bold',
        }}>
            {text}
        </div>
    );
};

// ===================================
// Types
// ===================================

interface Scene {
    order: number;
    url: string;
    type: 'video' | 'image';
    durationInSeconds?: number;
}

interface SubtitleWord {
    word: string;
    start: number; // seconds
    end: number;   // seconds
}

export interface ViralShortProps {
    audioUrl: string;
    subtitles?: SubtitleWord[];
    scenes: Scene[];
    avatarVideoUrl?: string; // Lip-synced avatar
    metadata?: {
        topic: string;
        niche: string;
    };
}

// ===================================
// Main Component
// ===================================

export const ViralShort: React.FC<ViralShortProps> = ({
    audioUrl,
    subtitles = [],
    scenes,
    avatarVideoUrl,
    metadata
}) => {
    const frame = useCurrentFrame();
    const { fps, durationInFrames } = useVideoConfig();

    // Calculate current time in seconds
    const currentTime = frame / fps;

    // Find current subtitle word
    const currentWord = subtitles.find(
        word => currentTime >= word.start && currentTime <= word.end
    );

    // Calculate scene durations (default 5s if missing)
    // In a real app, this logic handles transitions more gracefully
    const defaultSceneDuration = 5 * fps;
    let currentFrameCount = 0;

    return (
        <AbsoluteFill style={{ backgroundColor: 'black' }}>
            {/* 1. Background Visual Scenes */}
            <AbsoluteFill>
                {scenes.map((scene, index) => {
                    const sceneDuration = (scene.durationInSeconds || 5) * fps;
                    const startTime = currentFrameCount;
                    currentFrameCount += sceneDuration;

                    return (
                        <Sequence key={index} from={startTime} durationInFrames={Math.ceil(sceneDuration)}>
                            {scene.type === 'video' ? (
                                <Video src={scene.url} />
                            ) : (
                                <Img src={scene.url} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                            )}
                        </Sequence>
                    );
                })}
            </AbsoluteFill>

            {/* 2. Avatar Overlay (Circle Bottom Right) */}
            {avatarVideoUrl && (
                <AbsoluteFill>
                    <div style={{
                        position: 'absolute',
                        bottom: 50,
                        right: 50,
                        width: 300,
                        height: 300,
                        borderRadius: '50%',
                        overflow: 'hidden',
                        border: '5px solid white',
                        boxShadow: '0 0 20px rgba(0,0,0,0.5)',
                    }}>
                        <Video
                            src={avatarVideoUrl}
                            style={{
                                width: '100%',
                                height: '100%',
                                objectFit: 'cover'
                            }}
                        />
                    </div>
                </AbsoluteFill>
            )}

            {/* 3. Text Overlay / Captions */}
            {currentWord && (
                <Subtitle text={currentWord.word} />
            )}

            {/* 4. Audio Track */}
            <Audio src={audioUrl} />
        </AbsoluteFill>
    );
};

// Helper for video tag since Remotion's Video tag might act differently in preview vs render?
// Actually 'remotion' exports 'Video' component which wraps HTML5 video.
import { Video } from 'remotion';
