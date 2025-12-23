
import express from 'express';
import { bundle } from '@remotion/bundler';
import { renderMedia, selectComposition } from '@remotion/renderer';
import path from 'path';
import fs from 'fs';
import os from 'os';

const app = express();
app.use(express.json());

const PORT = process.env.PORT || 8000;

// Cache the bundle to avoid rebuilding on every request
let bundleLocation: string | null = null;

const getBundleLocation = async () => {
    if (bundleLocation) return bundleLocation;
    console.log('Bundling Remotion project...');
    bundleLocation = await bundle({
        entryPoint: path.join(__dirname, 'index.ts'),
        webpackOverride: (config) => config, // Default
    });
    console.log('Bundling complete:', bundleLocation);
    return bundleLocation;
};

// Helper middleware for basic logging
app.use((req, res, next) => {
    console.log(`[${new Date().toISOString()}] ${req.method} ${req.url}`);
    next();
});

// Render endpoint
app.post('/render', async (req, res) => {
    try {
        const { compositionId, inputProps, codec = 'h264', imageFormat = 'jpeg' } = req.body;

        if (!compositionId) {
            return res.status(400).json({ error: 'Missing compositionId' });
        }

        const bundled = await getBundleLocation();

        // 1. Get composition details to determine duration (if dynamic duration needed)
        const composition = await selectComposition({
            serveUrl: bundled,
            id: compositionId,
            inputProps,
        });

        // 2. Determine output path
        const tmpDir = os.tmpdir();
        const renderId = `render_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        const outputFileName = `${renderId}.mp4`;
        const outputPath = path.join(tmpDir, outputFileName);

        // 3. Start rendering (in-process for simplicity in this MVP)
        // In a real scaler, you might spin up a Lambda or worker.
        console.log(`Starting render: ${renderId} (${composition.durationInFrames} frames)`);

        // Note: For long renders, this will block the request if we await it.
        // For now, we await to keep it simple and return the result synchronously.
        await renderMedia({
            composition,
            serveUrl: bundled,
            codec: codec as any,
            outputLocation: outputPath,
            inputProps,
            imageFormat: imageFormat as any,
        });

        console.log(`Render complete: ${outputPath}`);

        // 4. Return result (In prod, upload to R2 here and return URL)
        // For this demo, we'll assume we can serve it back or upload it.
        // Let's assume we upload it or return a file path the agent can grab if local.
        // Since the agent is in another pod, we must upload to R2 or share a volume.
        // For MVP, enable downloading the file via GET endpoint.

        const downloadUrl = `http://${req.hostname}:${PORT}/download/${outputFileName}`;

        res.json({
            status: 'done',
            renderId,
            outputFile: outputPath,
            outputUrl: downloadUrl // Mock URL, in reality use R2
        });

    } catch (error) {
        console.error('Render error:', error);
        res.status(500).json({
            status: 'error',
            error: error instanceof Error ? error.message : String(error)
        });
    }
});

// Download endpoint for retrieving local renders
app.get('/download/:filename', (req, res) => {
    const filePath = path.join(os.tmpdir(), req.params.filename);
    if (fs.existsSync(filePath)) {
        res.sendFile(filePath);
    } else {
        res.status(404).send('File not found');
    }
});

// Status check (for async polling support later)
app.get('/render/:renderId/status', (req, res) => {
    // Implementing state persistence is out of scope for MVP
    // Assuming sync rendering for now.
    res.json({ status: 'unknown' });
});

app.listen(PORT, () => {
    console.log(`Remotion SSR server listening on port ${PORT}`);
});
