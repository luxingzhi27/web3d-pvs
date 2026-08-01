const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');

const app = express();
const PORT = 8050;

// Enable CORS so the React/Three.js frontend can easily send requests
app.use(cors());

// Increase payload limit because trajectory JSON files can be large after 10 mins
app.use(express.json({ limit: '50mb' }));

// Set up the directory to store trajectories
const SAVE_DIR = path.join(__dirname, '..', 'assets', 'trajectories');

// Ensure the directory exists
if (!fs.existsSync(SAVE_DIR)) {
    fs.mkdirSync(SAVE_DIR, { recursive: true });
    console.log(`Created trajectory directory at: ${SAVE_DIR}`);
}

app.post('/api/trajectory/save', (req, res) => {
    try {
        const payload = req.body;
        
        if (!payload || !payload.csvData) {
            return res.status(400).json({ error: 'Invalid or empty trajectory payload' });
        }

        // Generate a formatted timestamp filename
        const date = new Date();
        const yyyy = date.getFullYear();
        const mm = String(date.getMonth() + 1).padStart(2, '0');
        const dd = String(date.getDate()).padStart(2, '0');
        const hh = String(date.getHours()).padStart(2, '0');
        const min = String(date.getMinutes()).padStart(2, '0');
        const ss = String(date.getSeconds()).padStart(2, '0');
        
        const filename = `trajectory_${yyyy}${mm}${dd}_${hh}${min}${ss}.csv`;
        const filepath = path.join(SAVE_DIR, filename);

        // Write to file natively asynchronously
        fs.writeFile(filepath, payload.csvData, (err) => {
            if (err) {
                console.error(`Failed to save trajectory to ${filepath}:`, err);
                return res.status(500).json({ error: 'Failed to write file to disk' });
            }
            
            console.log(`[+] Successfully saved trajectory: ${filename}`);
            res.status(200).json({ message: 'Trajectory saved successfully', file: filename });
        });
        
    } catch (err) {
        console.error('Error processing trajectory data:', err);
        res.status(500).json({ error: 'Internal server error' });
    }
});

const CUSTOM_STARTS_FILE = path.join(__dirname, '../../slm2viewer/assets/custom_starts.json');

app.post('/api/starts/save', (req, res) => {
    try {
        const payload = req.body;
        if (!payload || !payload.position || !payload.target) {
            return res.status(400).json({ error: 'Invalid start point payload' });
        }
        
        let starts = [];
        if (fs.existsSync(CUSTOM_STARTS_FILE)) {
            const data = fs.readFileSync(CUSTOM_STARTS_FILE, 'utf8');
            try {
                starts = JSON.parse(data);
            } catch (e) { starts = []; }
        }
        
        starts.push(payload);
        fs.writeFileSync(CUSTOM_STARTS_FILE, JSON.stringify(starts, null, 2));
        
        console.log(`[+] Saved custom start point to custom_starts.json`);
        res.status(200).json({ message: 'Start point saved' });
    } catch (err) {
        console.error('Error saving start point:', err);
        res.status(500).json({ error: 'Internal server error' });
    }
});

app.listen(PORT, () => {
    console.log(`Trajectory Collection Server running on http://localhost:${PORT}`);
    console.log(`Saving files to: ${SAVE_DIR}`);
});
