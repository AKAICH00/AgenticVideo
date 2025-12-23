/**
 * YouTubeVideo - Horizontal 16:9 composition for YouTube
 */

import React from 'react';
import {
    AbsoluteFill,
    Audio,
    Img,
    Sequence,
    Video,
    useCurrentFrame,
    useVideoConfig,
    interpolate,
    spring,
} from 'remotion';

interface Scene {
    order: number;
    url: string;
    type: 'video' | 'image';
    durationInSeconds?: number;
}

interface SubtitleWord {
    word: string;
    start: number;
    end: number;
}

export interface YouTubeVideoProps {
    audioUrl: string;
    subtitles?: SubtitleWord[];
    scenes: Scene[];
    title?: string;
    introUrl?: string;
    outroUrl?: string;
    metadata?: {
        topic: string;
        niche: string;
    };
}

const LowerThird: React.FC<{ title: string }> = ({ title }) => {
    const frame = useCurrentFrame();
    const { fps } = useVideoConfig();
    const slideIn = spring({ fps, frame, config: { damping: 50 } });

    return (
        <div style={{
            position: 'absolute',
            bottom: 80,
            left: 0,
            transform: \`translateX(\${interpolate(slideIn, [0, 1], [-400, 0])}px)\`,
        }}>
            <div style={{
                backgroundColor: 'rgba(0, 0, 0, 0.85)',
                padding: '15px 30px',
                borderLeft: '4px solid #ff0000',
            }}>
                <div style={{
                    fontSize: 28,
                    color: 'white',
                    fontFamily: 'Helvetica, Arial, sans-serif',
                    fontWeight: 'bold',
                }}>
                    {title}
                </div>
            </div>
        </div>
    );
};

const Subtitle: React.FC<{ text: string }> = ({ text }) => (
    <div style={{
        position: 'absolute',
        bottom: 50,
        width: '100%',
        textAlign: 'center',
        fontSize: 42,
        color: 'white',
        textShadow: '2px 2px 4px rgba(0,0,0,0.9)',
        fontFamily: 'Helvetica, Arial, sans-serif',
        fontWeight: 'bold',
        padding: '0 100px',
    }}>
        {text}
    </div>
);

const IntroSequence: React.FC<{ title: string }> = ({ title }) => {
    const frame = useCurrentFrame();
    const { fps } = useVideoConfig();
    const opacity = interpolate(frame, [0, fps * 0.5, fps * 4.5, fps * 5], [0, 1, 1, 0]);
    const scale = spring({ fps, frame, config: { damping: 100 } });

    return (
        <AbsoluteFill style={{
            backgroundColor: '#0a0a0a',
            justifyContent: 'center',
            alignItems: 'center',
            opacity,
        }}>
            <div style={{
                fontSize: 72,
                color: 'white',
                fontFamily: 'Helvetica, Arial, sans-serif',
                fontWeight: 'bold',
                textAlign: 'center',
                transform: \`scale(\${scale})\`,
                maxWidth: '80%',
            }}>
                {title}
            </div>
        </AbsoluteFill>
    );
};

export const YouTubeVideo: React.FC<YouTubeVideoProps> = ({
    audioUrl,
    subtitles = [],
    scenes,
    title,
    introUrl,
    outroUrl,
}) => {
    const frame = useCurrentFrame();
    const { fps, durationInFrames } = useVideoConfig();
    const currentTime = frame / fps;
    const currentWord = subtitles.find(w => currentTime >= w.start && currentTime <= w.end);
    const introFrames = introUrl ? 5 * fps : 0;
    const outroFrames = outroUrl ? 5 * fps : 0;
    let currentFrameCount = introFrames;

    return (
        <AbsoluteFill style={{ backgroundColor: 'black' }}>
            {introUrl ? (
                <Sequence from={0} durationInFrames={introFrames}>
                    <Video src={introUrl} />
                </Sequence>
            ) : title ? (
                <Sequence from={0} durationInFrames={5 * fps}>
                    <IntroSequence title={title} />
                </Sequence>
            ) : null}

            <AbsoluteFill>
                {scenes.map((scene, index) => {
                    const sceneDuration = (scene.durationInSeconds || 5) * fps;
                    const startFrame = currentFrameCount;
                    currentFrameCount += sceneDuration;
                    return (
                        <Sequence key={index} from={startFrame} durationInFrames={Math.ceil(sceneDuration)}>
                            {scene.type === 'video' ? (
                                <Video src={scene.url} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                            ) : (
                                <Img src={scene.url} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                            )}
                        </Sequence>
                    );
                })}
            </AbsoluteFill>

            {outroUrl && (
                <Sequence from={durationInFrames - outroFrames} durationInFrames={outroFrames}>
                    <Video src={outroUrl} />
                </Sequence>
            )}

            {title && frame >= introFrames && frame < introFrames + 5 * fps && (
                <Sequence from={introFrames} durationInFrames={5 * fps}>
                    <LowerThird title={title} />
                </Sequence>
            )}

            {currentWord && <Subtitle text={currentWord.word} />}
            {audioUrl && <Audio src={audioUrl} />}
        </AbsoluteFill>
    );
};
