/**
 * Remotion Root - Main entry point for all compositions
 *
 * Registers compositions for:
 * - ViralShort: Vertical 9:16 format (TikTok, Shorts, Reels)
 * - YouTubeVideo: Horizontal 16:9 format (YouTube)
 */

import { Composition } from 'remotion';
import { ViralShort, ViralShortProps } from './ViralShort';
import { YouTubeVideo, YouTubeVideoProps } from './YouTubeVideo';
import { z } from 'zod';

// ===================================
// Schema Definitions (for type-safe props)
// ===================================

const SceneSchema = z.object({
    order: z.number(),
    url: z.string(),
    type: z.enum(['video', 'image']),
    durationInSeconds: z.number().optional().default(5),
});

const SubtitleSchema = z.object({
    word: z.string(),
    start: z.number(),
    end: z.number(),
});

const ViralShortSchema = z.object({
    audioUrl: z.string(),
    subtitles: z.array(SubtitleSchema).optional().default([]),
    scenes: z.array(SceneSchema),
    avatarVideoUrl: z.string().optional(),
    metadata: z.object({
        topic: z.string(),
        niche: z.string(),
    }).optional(),
});

const YouTubeVideoSchema = z.object({
    audioUrl: z.string(),
    subtitles: z.array(SubtitleSchema).optional().default([]),
    scenes: z.array(SceneSchema),
    title: z.string().optional(),
    introUrl: z.string().optional(),
    outroUrl: z.string().optional(),
    metadata: z.object({
        topic: z.string(),
        niche: z.string(),
    }).optional(),
});

// ===================================
// Composition Registration
// ===================================

export const RemotionRoot: React.FC = () => {
    return (
        <>
            {/* Vertical Short-Form Video (9:16) */}
            <Composition
                id="ViralShort"
                component={ViralShort}
                durationInFrames={30 * 60}  // 60 seconds at 30fps (max, actual from props)
                fps={30}
                width={1080}
                height={1920}
                schema={ViralShortSchema}
                defaultProps={{
                    audioUrl: '',
                    subtitles: [],
                    scenes: [],
                    avatarVideoUrl: undefined,
                    metadata: { topic: '', niche: '' },
                }}
                calculateMetadata={({ props }) => {
                    // Calculate actual duration from scenes
                    const totalSeconds = props.scenes.reduce(
                        (sum, scene) => sum + (scene.durationInSeconds || 5),
                        0
                    );
                    return {
                        durationInFrames: Math.ceil(totalSeconds * 30),
                    };
                }}
            />

            {/* Horizontal Long-Form Video (16:9) */}
            <Composition
                id="YouTubeVideo"
                component={YouTubeVideo}
                durationInFrames={30 * 600}  // 10 minutes max at 30fps
                fps={30}
                width={1920}
                height={1080}
                schema={YouTubeVideoSchema}
                defaultProps={{
                    audioUrl: '',
                    subtitles: [],
                    scenes: [],
                    title: undefined,
                    introUrl: undefined,
                    outroUrl: undefined,
                    metadata: { topic: '', niche: '' },
                }}
                calculateMetadata={({ props }) => {
                    const totalSeconds = props.scenes.reduce(
                        (sum, scene) => sum + (scene.durationInSeconds || 5),
                        0
                    );
                    // Add 5s for intro + 5s for outro if present
                    const introSeconds = props.introUrl ? 5 : 0;
                    const outroSeconds = props.outroUrl ? 5 : 0;
                    return {
                        durationInFrames: Math.ceil((totalSeconds + introSeconds + outroSeconds) * 30),
                    };
                }}
            />
        </>
    );
};
