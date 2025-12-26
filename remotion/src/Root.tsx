/**
 * Remotion Root - Main entry point for all compositions
 *
 * Registers compositions for:
 * - ViralShort: Vertical 9:16 format (TikTok, Shorts, Reels)
 * - YouTubeVideo: Horizontal 16:9 format (YouTube)
 */

import React from 'react';
import { Composition } from 'remotion';
import { ViralShort, ViralShortProps } from './ViralShort';
import { YouTubeVideo, YouTubeVideoProps } from './YouTubeVideo';

// ===================================
// Default Props
// ===================================

const defaultViralShortProps: ViralShortProps = {
    audioUrl: '',
    subtitles: [],
    scenes: [],
    avatarVideoUrl: undefined,
    metadata: { topic: '', niche: '' },
};

const defaultYouTubeVideoProps: YouTubeVideoProps = {
    audioUrl: '',
    subtitles: [],
    scenes: [],
    title: undefined,
    introUrl: undefined,
    outroUrl: undefined,
    metadata: { topic: '', niche: '' },
};

// ===================================
// Composition Registration
// ===================================

export const RemotionRoot: React.FC = () => {
    return (
        <>
            {/* Vertical Short-Form Video (9:16) */}
            <Composition
                id="ViralShort"
                component={ViralShort as any}
                durationInFrames={30 * 60}  // 60 seconds at 30fps (max, actual from props)
                fps={30}
                width={1080}
                height={1920}
                defaultProps={defaultViralShortProps as any}
                calculateMetadata={(({ props }: { props: ViralShortProps }) => {
                    // Calculate actual duration from scenes
                    const totalSeconds = props.scenes.reduce(
                        (sum, scene) => sum + (scene.durationInSeconds || 5),
                        0
                    );
                    return {
                        durationInFrames: Math.max(30, Math.ceil(totalSeconds * 30)),
                    };
                }) as any}
            />

            {/* Horizontal Long-Form Video (16:9) */}
            <Composition
                id="YouTubeVideo"
                component={YouTubeVideo as any}
                durationInFrames={30 * 600}  // 10 minutes max at 30fps
                fps={30}
                width={1920}
                height={1080}
                defaultProps={defaultYouTubeVideoProps as any}
                calculateMetadata={(({ props }: { props: YouTubeVideoProps }) => {
                    const totalSeconds = props.scenes.reduce(
                        (sum, scene) => sum + (scene.durationInSeconds || 5),
                        0
                    );
                    // Add 5s for intro + 5s for outro if present
                    const introSeconds = props.introUrl ? 5 : 0;
                    const outroSeconds = props.outroUrl ? 5 : 0;
                    return {
                        durationInFrames: Math.max(30, Math.ceil((totalSeconds + introSeconds + outroSeconds) * 30)),
                    };
                }) as any}
            />
        </>
    );
};
