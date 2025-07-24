#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TF1.h>
#include <TCanvas.h>
#include <TSystem.h>
#include <TMath.h>
#include <TStyle.h>
#include <TLegend.h>
#include <TPaveStats.h>
#include <TGaxis.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <string>
#include <map>
#include <set>
#include <sys/stat.h>
#include <unistd.h>
#include <ctime>
#include <TGraph.h>
#include <TGraphErrors.h>

using std::cout;
using std::endl;
using namespace std;

// Constants
const int N_PMTS = 12;
const int PMT_CHANNEL_MAP[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
const int PULSE_THRESHOLD = 30;     // ADC threshold for pulse detection
const int BS_UNCERTAINTY = 5;       // Baseline uncertainty (ADC)
const int EV61_THRESHOLD = 1200;    // Beam on if channel 22 > this (ADC)
const double MUON_ENERGY_THRESHOLD = 50; // Min PMT energy for muon (p.e.)
const double MICHEL_ENERGY_MIN = 40;    // Min PMT energy for Michel (p.e.)
const double MICHEL_ENERGY_MAX = 1000;  // Max PMT energy for Michel (p.e.)
const double MICHEL_ENERGY_MAX_DT = 400; // Max PMT energy for dt plots (p.e.)
const double MICHEL_DT_MIN = 0.76;       // Min time after muon for Michel (µs)
const double MICHEL_DT_MAX = 16.0;      // Max time after muon for Michel (µs)
const int ADCSIZE = 45;                 // Number of ADC samples per waveform

// Generate unique output directory with timestamp
string getTimestamp() {
    time_t now = time(nullptr);
    struct tm *t = localtime(&now);
    char buffer[20];
    strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", t);
    return string(buffer);
}
const string OUTPUT_DIR = "./AnalysisOutput_" + getTimestamp();

const std::vector<double> SIDE_VP_THRESHOLDS = {750, 950, 1200, 1375, 525, 700, 700, 500}; // Channels 12-19 (ADC)
const double TOP_VP_THRESHOLD = 450; // Channels 20-21 (ADC)
const double FIT_MIN = 1.0; // Fit range min (µs)
const double FIT_MAX = 16.0; // Fit range max (µs)

// Pulse structure
struct pulse {
    double start;          // Start time (µs)
    double end;            // End time (µs)
    double peak;           // Max amplitude (p.e. for PMTs, ADC for SiPMs)
    double energy;         // Energy (p.e. for PMTs, ADC for SiPMs)
    double number;         // Number of channels with pulse
    bool single;           // Timing consistency
    bool beam;             // Beam status
    double trigger;        // Trigger type
    double side_vp_energy; // Side veto energy (ADC)
    double top_vp_energy;  // Top veto energy (ADC)
    double all_vp_energy;  // All veto energy (ADC)
    double last_muon_time; // Time of last muon (µs)
    bool is_muon;          // Muon candidate flag
    bool is_michel;        // Michel electron candidate flag
};

// Temporary pulse structure
struct pulse_temp {
    double start;  // Start time (µs)
    double end;    // End time (µs)
    double peak;   // Max amplitude
    double energy; // Energy
};

// SPE fitting function
Double_t SPEfit(Double_t *x, Double_t *par) {
    Double_t term1 = par[0] * exp(-0.5 * pow((x[0] - par[1]) / par[2], 2));
    Double_t term2 = par[3] * exp(-0.5 * pow((x[0] - par[4]) / par[5], 2));
    Double_t term3 = par[6] * exp(-0.5 * pow((x[0] - sqrt(2) * par[4]) / sqrt(2 * pow(par[5], 2) - pow(par[2], 2)), 2));
    Double_t term4 = par[7] * exp(-0.5 * pow((x[0] - sqrt(3) * par[4]) / sqrt(3 * pow(par[5], 2) - 2 * pow(par[2], 2)), 2));
    return term1 + term2 + term3 + term4;
}

// Double exponential fit function with fixed accidental background
double tau_accidental = 0.0; // Global accidental lifetime
Double_t ConstrainedDoubleExpFit(Double_t *x, Double_t *par) {
    return par[0] * exp(-x[0]/par[1]) + par[2] * exp(-x[0]/tau_accidental);
}

// Utility functions
template<typename T>
double getAverage(const std::vector<T>& v) {
    if (v.empty()) return 0;
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

template<typename T>
double mostFrequent(const std::vector<T>& v) {
    if (v.empty()) return 0;
    std::map<T, int> count;
    for (const auto& val : v) count[val]++;
    T most_common = v[0];
    int max_count = 0;
    for (const auto& pair : count) {
        if (pair.second > max_count) {
            max_count = pair.second;
            most_common = pair.first;
        }
    }
    return max_count > 1 ? most_common : getAverage(v);
}

template<typename T>
double variance(const std::vector<T>& v) {
    if (v.size() <= 1) return 0;
    double mean = getAverage(v);
    double sum = 0;
    for (const auto& val : v) {
        sum += (val - mean) * (val - mean);
    }
    return sum / (v.size() - 1);
}

// Create output directory
void createOutputDirectory(const string& dirName) {
    struct stat st;
    if (stat(dirName.c_str(), &st) != 0) {
        if (mkdir(dirName.c_str(), 0755) != 0) {
            cerr << "Error: Could not create directory " << dirName << endl;
            exit(1);
        }
        cout << "Created output directory: " << dirName << endl;
    } else {
        cout << "Output directory already exists: " << dirName << endl;
    }
}

// SPE calibration function
void performCalibration(const string &calibFileName, Double_t *mu1, Double_t *mu1_err) {
    TFile *calibFile = TFile::Open(calibFileName.c_str());
    if (!calibFile || calibFile->IsZombie()) {
        cerr << "Error opening calibration file: " << calibFileName << endl;
        exit(1);
    }

    TTree *calibTree = (TTree*)calibFile->Get("tree");
    if (!calibTree) {
        cerr << "Error accessing tree in calibration file" << endl;
        calibFile->Close();
        exit(1);
    }

    TCanvas *c = new TCanvas("c", "SPE Fits", 800, 600);
    TH1F *histArea[N_PMTS];
    Long64_t nLEDFlashes[N_PMTS] = {0};
    for (int i = 0; i < N_PMTS; i++) {
        histArea[i] = new TH1F(Form("PMT%d_Area", i + 1),
                               Form("PMT %d;ADC Counts;Events", i + 1), 150, -50, 400);
    }

    Int_t triggerBits;
    Double_t area[23];
    calibTree->SetBranchAddress("triggerBits", &triggerBits);
    calibTree->SetBranchAddress("area", area);

    Long64_t nEntries = calibTree->GetEntries();
    cout << "Processing " << nEntries << " calibration events from " << calibFileName << "..." << endl;

    for (Long64_t entry = 0; entry < nEntries; entry++) {
        calibTree->GetEntry(entry);
        if (triggerBits != 16) continue;
        for (int pmt = 0; pmt < N_PMTS; pmt++) {
            histArea[pmt]->Fill(area[PMT_CHANNEL_MAP[pmt]]);
            nLEDFlashes[pmt]++;
        }
    }

    for (int i = 0; i < N_PMTS; i++) {
        if (histArea[i]->GetEntries() < 1000) {
            cerr << "Warning: Insufficient data for PMT " << i + 1 << " in " << calibFileName << endl;
            mu1[i] = 0;
            mu1_err[i] = 0;
            delete histArea[i];
            continue;
        }

        TF1 *fitFunc = new TF1("fitFunc", SPEfit, -50, 400, 8);
        Double_t histMean = histArea[i]->GetMean();
        Double_t histRMS = histArea[i]->GetRMS();

        fitFunc->SetParameters(1000, histMean - histRMS, histRMS / 2,
                              1000, histMean, histRMS,
                              500, 200);

        histArea[i]->Fit(fitFunc, "Q", "", -50, 400);

        mu1[i] = fitFunc->GetParameter(4);
        Double_t sigma_mu1 = fitFunc->GetParError(4);
        Double_t sigma1 = fitFunc->GetParameter(5);
        mu1_err[i] = sqrt(pow(sigma_mu1, 2) + pow(sigma1 / sqrt(nLEDFlashes[i]), 2));

        // Plot SPE fit
        c->Clear();
        histArea[i]->Draw();
        fitFunc->Draw("same");
        TLegend *leg = new TLegend(0.6, 0.7, 0.9, 0.9);
        leg->AddEntry(histArea[i], Form("PMT %d Data", i + 1), "l");
        leg->AddEntry(fitFunc, "SPE Fit", "l");
        leg->AddEntry((TObject*)0, Form("mu1 = %.2f #pm %.2f", mu1[i], mu1_err[i]), "");
        leg->Draw();
        string plotName = OUTPUT_DIR + Form("/SPE_Fit_PMT%d.png", i + 1);
        c->Update();
        c->SaveAs(plotName.c_str());
        cout << "Saved SPE plot: " << plotName << endl;
        delete leg;
        delete fitFunc;
        delete histArea[i];
    }

    delete c;
    calibFile->Close();
}

int main(int argc, char *argv[]) {
    // Parse command-line arguments
    if (argc < 3) {
        cout << "Usage: " << argv[0] << " <calibration_file> <input_file1> [<input_file2> ...]" << endl;
        return -1;
    }

    string calibFileName = argv[1];
    vector<string> inputFiles;
    for (int i = 2; i < argc; i++) {
        inputFiles.push_back(argv[i]);
    }

    // Create output directory
    createOutputDirectory(OUTPUT_DIR);

    cout << "Calibration file: " << calibFileName << endl;
    cout << "Input files:" << endl;
    for (const auto& file : inputFiles) {
        cout << "  " << file << endl;
    }

    // Check if calibration file exists
    if (gSystem->AccessPathName(calibFileName.c_str())) {
        cerr << "Error: Calibration file " << calibFileName << " not found" << endl;
        return -1;
    }

    // Check if at least one input file exists
    bool anyInputFileExists = false;
    for (const auto& file : inputFiles) {
        if (!gSystem->AccessPathName(file.c_str())) {
            anyInputFileExists = true;
            break;
        }
    }
    if (!anyInputFileExists) {
        cerr << "Error: No input files found" << endl;
        return -1;
    }

    // Perform SPE calibration
    Double_t mu1[N_PMTS] = {0};
    Double_t mu1_err[N_PMTS] = {0};
    performCalibration(calibFileName, mu1, mu1_err);

    // Print calibration results
    cout << "SPE Calibration Results (from " << calibFileName << "):\n";
    for (int i = 0; i < N_PMTS; i++) {
        cout << "PMT " << i + 1 << ": mu1 = " << mu1[i] << " ± " << mu1_err[i] << " ADC counts/p.e.\n";
    }

    // Statistics counters
    int num_muons = 0;
    int num_michels = 0;
    int num_events = 0;

    // Map to track triggerBits counts
    std::map<int, int> trigger_counts;

    // Define histograms
    TH1D* h_muon_energy = new TH1D("muon_energy", "Muon Energy Distribution (with Michel Electrons);Energy (p.e.);Counts/100 p.e.", 550, -500, 5000);
    TH1D* h_michel_energy = new TH1D("michel_energy", "Michel Electron Energy Distribution;Energy (p.e.);Counts/8 p.e.", 100, 0, 800);
    TH1D* h_dt_michel = new TH1D("DeltaT", "Muon-Michel Time Difference ;Time to Previous event(Muon)(#mus);Counts/0.08 #mus", 200, 0, MICHEL_DT_MAX);
    TH2D* h_energy_vs_dt = new TH2D("energy_vs_dt", "Michel Energy vs Time Difference;dt (#mus);Energy (p.e.)", 160, 0, 16, 200, 0, 1000);
    TH1D* h_side_vp_muon = new TH1D("side_vp_muon", "Side Veto Energy for Muons;Energy (ADC);Counts", 200, 0, 5000);
    TH1D* h_top_vp_muon = new TH1D("top_vp_muon", "Top Veto Energy for Muons;Energy (ADC);Counts", 200, 0, 1000);
    TH1D* h_trigger_bits = new TH1D("trigger_bits", "Trigger Bits Distribution;Trigger Bits;Counts", 36, 0, 36);
    
    // New histograms for accidental events
    TH1D* h_dt_accidental = new TH1D("dt_accidental", "Accidental Events Time Difference;dt (#mus);Counts", 200, 0, MICHEL_DT_MAX);
    TH1D* h_accidental_energy = new TH1D("accidental_energy", "Accidental Events Energy Distribution;Energy (p.e.);Counts", 100, 0, 800);
    std::vector<double> accidental_times;

    for (const auto& inputFileName : inputFiles) {
        // Check if input file exists
        if (gSystem->AccessPathName(inputFileName.c_str())) {
            cout << "Could not open file: " << inputFileName << ". Skipping..." << endl;
            continue;
        }

        TFile *f = new TFile(inputFileName.c_str());
        if (!f || f->IsZombie()) {
            cout << "Could not open file: " << inputFileName << endl;
            continue;
        }

        cout << "Processing file: " << inputFileName << endl;

        TTree* t = (TTree*)f->Get("tree");
        if (!t) {
            cout << "Could not find tree in file: " << inputFileName << endl;
            f->Close();
            continue;
        }

        // Declaration of leaf types
        Int_t eventID;
        Int_t nSamples[23];
        Short_t adcVal[23][45];
        Double_t baselineMean[23];
        Double_t baselineRMS[23];
        Double_t pulseH[23];
        Int_t peakPosition[23];
        Double_t area[23];
        Long64_t nsTime;
        Int_t triggerBits;

        // Set branch addresses
        t->SetBranchAddress("eventID", &eventID);
        t->SetBranchAddress("nSamples", nSamples);
        t->SetBranchAddress("adcVal", adcVal);
        t->SetBranchAddress("baselineMean", baselineMean);
        t->SetBranchAddress("baselineRMS", baselineRMS);
        t->SetBranchAddress("pulseH", pulseH);
        t->SetBranchAddress("peakPosition", peakPosition);
        t->SetBranchAddress("area", area);
        t->SetBranchAddress("nsTime", &nsTime);
        t->SetBranchAddress("triggerBits", &triggerBits);

        int numEntries = t->GetEntries();
        cout << "Processing " << numEntries << " entries in " << inputFileName << endl;
        double last_muon_time = 0.0;
        double last_accidental_time = 0.0; // Track previous accidental candidate time
        std::set<double> michel_muon_times;
        std::vector<std::pair<double, double>> muon_candidates;

        // First pass: Identify Michel electrons and their muon times
        for (int iEnt = 0; iEnt < numEntries; iEnt++) {
            t->GetEntry(iEnt);
            num_events++;

            // Fill triggerBits histogram and track counts
            h_trigger_bits->Fill(triggerBits);
            trigger_counts[triggerBits]++;
            // Check for out-of-range triggerBits
            if (triggerBits < 0 || triggerBits > 36) {
                cout << "Warning: triggerBits = " << triggerBits << " out of histogram range (0–36) in file " << inputFileName << ", event " << eventID << endl;
            }

            // Initialize pulse
            struct pulse p;
            p.start = nsTime / 1000.0; // Convert ns to µs
            p.end = nsTime / 1000.0;
            p.peak = 0;
            p.energy = 0;
            p.number = 0;
            p.single = false;
            p.beam = false;
            p.trigger = triggerBits;
            p.side_vp_energy = 0;
            p.top_vp_energy = 0;
            p.all_vp_energy = 0;
            p.last_muon_time = last_muon_time;
            p.is_muon = false;
            p.is_michel = false;

            std::vector<double> all_chan_start, all_chan_end, all_chan_peak, all_chan_energy;
            std::vector<double> side_vp_energy, top_vp_energy;
            std::vector<double> chan_starts_no_outliers;
            TH1D h_wf("h_wf", "Waveform", ADCSIZE, 0, ADCSIZE);

            bool pulse_at_end = false;
            int pulse_at_end_count = 0;
            std::vector<double> veto_energies(10, 0); // Channels 12-21

            for (int iChan = 0; iChan < 23; iChan++) {
                // Fill waveform histogram
                for (int i = 0; i < ADCSIZE; i++) {
                    h_wf.SetBinContent(i + 1, adcVal[iChan][i] - baselineMean[iChan]);
                }

                // Check beam status (channel 22)
                if (iChan == 22) {
                    double ev61_energy = 0;
                    for (int iBin = 1; iBin <= ADCSIZE; iBin++) {
                        ev61_energy += h_wf.GetBinContent(iBin);
                    }
                    if (ev61_energy > EV61_THRESHOLD) {
                        p.beam = true;
                    }
                }

                // Pulse detection
                std::vector<pulse_temp> pulses_temp;
                bool onPulse = false;
                int thresholdBin = 0, peakBin = 0;
                double peak = 0, pulseEnergy = 0;
                double allPulseEnergy = 0;

                for (int iBin = 1; iBin <= ADCSIZE; iBin++) {
                    double iBinContent = h_wf.GetBinContent(iBin);
                    if (iBin > 15) allPulseEnergy += iBinContent;

                    if (!onPulse && iBinContent >= PULSE_THRESHOLD) {
                        onPulse = true;
                        thresholdBin = iBin;
                        peakBin = iBin;
                        peak = iBinContent;
                        pulseEnergy = iBinContent;
                    } else if (onPulse) {
                        pulseEnergy += iBinContent;
                        if (peak < iBinContent) {
                            peak = iBinContent;
                            peakBin = iBin;
                        }
                        if (iBinContent < BS_UNCERTAINTY || iBin == ADCSIZE) {
                            pulse_temp pt;
                            pt.start = thresholdBin * 16.0 / 1000.0; // Convert ns to µs
                            pt.peak = iChan <= 11 && mu1[iChan] > 0 ? peak / mu1[iChan] : peak;
                            pt.end = iBin * 16.0 / 1000.0;
                            for (int j = peakBin - 1; j >= 1 && h_wf.GetBinContent(j) > BS_UNCERTAINTY; j--) {
                                if (h_wf.GetBinContent(j) > peak * 0.1) {
                                    pt.start = j * 16.0 / 1000.0;
                                }
                                pulseEnergy += h_wf.GetBinContent(j);
                            }
                            if (iChan <= 11) {
                                pt.energy = mu1[iChan] > 0 ? pulseEnergy / mu1[iChan] : 0;
                                all_chan_start.push_back(pt.start);
                                all_chan_end.push_back(pt.end);
                                all_chan_peak.push_back(pt.peak);
                                all_chan_energy.push_back(pt.energy);
                                if (pt.energy > 1) p.number += 1;
                            }
                            pulses_temp.push_back(pt);
                            peak = 0;
                            peakBin = 0;
                            pulseEnergy = 0;
                            thresholdBin = 0;
                            onPulse = false;
                        }
                    }
                }

                // Store energy for veto panels (ADC)
                if (iChan >= 12 && iChan <= 19) {
                    side_vp_energy.push_back(allPulseEnergy);
                    veto_energies[iChan - 12] = allPulseEnergy;
                } else if (iChan >= 20 && iChan <= 21) {
                    double factor = (iChan == 20) ? 1.07809 : 1.0;
                    top_vp_energy.push_back(allPulseEnergy * factor);
                    veto_energies[iChan - 12] = allPulseEnergy * factor;
                }

                // Check for pulses at waveform end
                if (iChan <= 11 && h_wf.GetBinContent(ADCSIZE) > 100) {
                    pulse_at_end_count++;
                    if (pulse_at_end_count >= 10) pulse_at_end = true;
                }

                h_wf.Reset();
            }

            // Aggregate pulse properties
            p.start += mostFrequent(all_chan_start);
            p.end += mostFrequent(all_chan_end);
            p.energy = std::accumulate(all_chan_energy.begin(), all_chan_energy.end(), 0.0);
            p.peak = std::accumulate(all_chan_peak.begin(), all_chan_peak.end(), 0.0);
            p.side_vp_energy = std::accumulate(side_vp_energy.begin(), side_vp_energy.end(), 0.0);
            p.top_vp_energy = std::accumulate(top_vp_energy.begin(), top_vp_energy.end(), 0.0);
            p.all_vp_energy = p.side_vp_energy + p.top_vp_energy;

            // Check timing consistency
            for (const auto& start : all_chan_start) {
                if (fabs(start - mostFrequent(all_chan_start)) < 10 * 16.0 / 1000.0) {
                    chan_starts_no_outliers.push_back(start);
                }
            }
            p.single = (variance(chan_starts_no_outliers) < 5 * 16.0 / 1000.0);

            // Muon detection
            bool veto_hit = false;
            for (size_t i = 0; i < SIDE_VP_THRESHOLDS.size(); i++) {
                if (veto_energies[i] > SIDE_VP_THRESHOLDS[i]) {
                    veto_hit = true;
                    break;
                }
            }
            if (!veto_hit && p.top_vp_energy > TOP_VP_THRESHOLD) veto_hit = true;

            if ((p.energy > MUON_ENERGY_THRESHOLD && veto_hit) ||
                (pulse_at_end && p.energy > MUON_ENERGY_THRESHOLD / 2 && veto_hit)) {
                p.is_muon = true;
                last_muon_time = p.start;
                num_muons++;
                muon_candidates.emplace_back(p.start, p.energy);
                h_side_vp_muon->Fill(p.side_vp_energy);
                h_top_vp_muon->Fill(p.top_vp_energy);
            }

            // Michel electron detection
            double dt = p.start - last_muon_time;
            bool veto_low = true;
            for (size_t i = 0; i < SIDE_VP_THRESHOLDS.size(); i++) {
                if (veto_energies[i] > SIDE_VP_THRESHOLDS[i]) {
                    veto_low = false;
                    break;
                }
            }
            if (veto_energies[8] > TOP_VP_THRESHOLD || veto_energies[9] > TOP_VP_THRESHOLD) {
                veto_low = false;
            }

            // Define common Michel electron criteria
            bool is_michel_candidate = p.energy >= MICHEL_ENERGY_MIN &&
                                      p.energy <= MICHEL_ENERGY_MAX &&
                                      dt >= MICHEL_DT_MIN &&
                                      dt <= MICHEL_DT_MAX &&
                                      p.number >= 8 &&
                                      veto_low &&
                                      p.trigger != 1 &&
                                      p.trigger != 4 &&
                                      p.trigger != 8 &&
                                      p.trigger != 16;
            h_energy_vs_dt->Fill(dt, p.energy);

            // Apply additional cut for dt and energy_vs_dt plots
            bool is_michel_for_dt = is_michel_candidate && p.energy <= MICHEL_ENERGY_MAX_DT;

            if (is_michel_candidate) {
                p.is_michel = true;
                num_michels++;
                michel_muon_times.insert(last_muon_time);
                // Fill Michel energy histogram with original criteria
                h_michel_energy->Fill(p.energy);
            }

            if (is_michel_for_dt) {
                // Fill dt and energy_vs_dt histograms with stricter energy cut
                h_dt_michel->Fill(dt);
            }
            
            // Collect accidental events (beam off, Michel selection without dt cut for collection)
            if (!p.beam) {  // Beam off
                // Apply Michel selection without dt cut for time collection
                if (p.energy >= MICHEL_ENERGY_MIN && p.energy <= MICHEL_ENERGY_MAX && 
                    p.number >= 8 && veto_low && 
                    p.trigger != 1 && p.trigger != 4 && p.trigger != 8 && p.trigger != 16) {
                    
                    // Calculate dt for accidental event
                    double acc_dt = p.start - last_accidental_time;
                    
                    // Fill accidental energy spectrum if dt is within 1.0–16.0 µs
                    if (acc_dt >= MICHEL_DT_MIN && acc_dt <= MICHEL_DT_MAX) {
                        h_accidental_energy->Fill(p.energy);
                    }
                    
                    // Store time for dt calculation
                    accidental_times.push_back(p.start);
                    last_accidental_time = p.start; // Update last accidental time
                }
            }

            p.last_muon_time = last_muon_time;
        }

        // Second pass: Fill h_muon_energy for muons associated with Michel electrons
        for (const auto& muon : muon_candidates) {
            if (michel_muon_times.find(muon.first) != michel_muon_times.end()) {
                h_muon_energy->Fill(muon.second);
            }
        }

        // Print stats to console
        cout << "File " << inputFileName << " Statistics:\n";
        cout << "Total Events: " << num_events << "\n";
        cout << "Muons Detected: " << num_muons << "\n";
        cout << "Michel Electrons Detected: " << num_michels << "\n";
        cout << "------------------------\n";

        f->Close();
        delete f;

        num_events = 0;
        num_muons = 0;
        num_michels = 0;
    }

    // Calculate time differences for accidental events (within 1.0–16.0 µs)
    if (!accidental_times.empty()) {
        // Sort times chronologically
        std::sort(accidental_times.begin(), accidental_times.end());
        
        // Calculate consecutive time differences
        for (size_t i = 1; i < accidental_times.size(); i++) {
            double dt = accidental_times[i] - accidental_times[i-1];
            if (dt >= MICHEL_DT_MIN && dt <= MICHEL_DT_MAX) {
                h_dt_accidental->Fill(dt);
            }
        }
    }

    // Print triggerBits distribution
    cout << "Trigger Bits Distribution (all files):\n";
    for (const auto& pair : trigger_counts) {
        cout << "Trigger " << pair.first << ": " << pair.second << " events\n";
    }
    cout << "------------------------\n";

    // Generate analysis plots
    TCanvas *c = new TCanvas("c", "Analysis Plots", 1200, 800);
    gStyle->SetOptStat(1111);
    gStyle->SetOptFit(1111);

    // Muon Energy
    c->Clear();
    h_muon_energy->SetLineColor(kBlue);
    h_muon_energy->Draw();
    c->Update();
    string plotName = OUTPUT_DIR + "/Muon_Energy.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    // Michel Energy
    c->Clear();
    h_michel_energy->SetLineColor(kRed);
    h_michel_energy->Draw();
    c->Update();
    plotName = OUTPUT_DIR + "/Michel_Energy.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    // Accidental Energy Spectrum
    c->Clear();
    h_accidental_energy->SetLineColor(kGreen);
    h_accidental_energy->SetLineWidth(2);
    h_accidental_energy->Draw();
    c->Update();
    plotName = OUTPUT_DIR + "/Accidental_Energy.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    // Accidental dt distribution and fit
    c->Clear();
    h_dt_accidental->SetLineWidth(2);
    h_dt_accidental->SetLineColor(kBlue);
    h_dt_accidental->GetXaxis()->SetTitle("Time Difference (#mus)");
    h_dt_accidental->GetYaxis()->SetTitle("Counts");
    h_dt_accidental->Draw("HIST");
    
    // Fit accidental dt with single exponential
    TF1* fAcc = nullptr;
    if (h_dt_accidental->GetEntries() > 10) {
        fAcc = new TF1("fAcc", "[0]*exp(-x/[1])", MICHEL_DT_MIN, MICHEL_DT_MAX);
        fAcc->SetParameters(h_dt_accidental->GetMaximum(), 10.0); // Initial guess
        fAcc->SetParNames("N", "#tau_{acc}");
        fAcc->SetLineColor(kRed);
        fAcc->SetLineWidth(2);
        
        h_dt_accidental->Fit(fAcc, "R");
        fAcc->Draw("SAME"); // Ensure fit line is drawn
        tau_accidental = fAcc->GetParameter(1); // Store accidental lifetime
        
        cout << "Accidental Lifetime Fit Results:\n";
        cout << "τ_acc = " << fAcc->GetParameter(1) << " ± " << fAcc->GetParError(1) << " µs\n";
        cout << "χ²/NDF = " << fAcc->GetChisquare() / fAcc->GetNDF() << endl;
        cout << "----------------------------------------" << endl;
    } else {
        cerr << "Warning: Insufficient accidental events for fit (" 
             << h_dt_accidental->GetEntries() << " events)" << endl;
        tau_accidental = 1.0; // Default value
    }
    
    c->Update();
    plotName = OUTPUT_DIR + "/Accidental_dt.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;
    if (fAcc) delete fAcc;
    
    // Michel dt with double exponential fit
    c->Clear();
    h_dt_michel->SetLineWidth(2);
    h_dt_michel->SetLineColor(kBlack);
    h_dt_michel->GetXaxis()->SetTitle("Time to previous event (Muon) [#mus]");
    h_dt_michel->Draw("HIST");

    TF1* doubleExpFit = nullptr;
    if (h_dt_michel->GetEntries() > 5) {
        // Create double exponential function with fixed accidental component
        doubleExpFit = new TF1("doubleExpFit", ConstrainedDoubleExpFit, FIT_MIN, FIT_MAX, 3);
        doubleExpFit->SetParNames("N_{#mu}", "#tau_{#mu}", "N_{acc}");
        
        // Set initial parameters
        double muon_lifetime_guess = 2.2;
        double N0_guess = h_dt_michel->GetBinContent(h_dt_michel->FindBin(FIT_MIN)) * 1.2;
        double N_acc_guess = h_dt_michel->GetBinContent(h_dt_michel->FindBin(FIT_MAX)) * 0.8;
        
        doubleExpFit->SetParameters(N0_guess, muon_lifetime_guess, N_acc_guess);
        doubleExpFit->SetParLimits(0, 0, N0_guess * 10);
        doubleExpFit->SetParLimits(1, 0.1, 20.0);
        doubleExpFit->SetParLimits(2, 0, N_acc_guess * 10);
        doubleExpFit->SetLineColor(kRed);
        doubleExpFit->SetLineWidth(3);
        
        // Perform fit
        int fitStatus = h_dt_michel->Fit(doubleExpFit, "RE+", "SAME", FIT_MIN, FIT_MAX);
        doubleExpFit->Draw("SAME"); // Ensure fit line is drawn
        
        // Update stats box
        gPad->Update();
        TPaveStats *stats = (TPaveStats*)h_dt_michel->FindObject("stats");
        if (stats) {
            stats->SetX1NDC(0.6);
            stats->SetX2NDC(0.9);
            stats->SetY1NDC(0.6);
            stats->SetY2NDC(0.9);
            stats->SetTextColor(kRed);
            stats->Clear();
            stats->AddText("Double Exponential Fit");
            stats->AddText(Form("#tau_{#mu} = %.4f #pm %.4f #mus", 
                                doubleExpFit->GetParameter(1), doubleExpFit->GetParError(1)));
            stats->AddText(Form("#chi^{2}/NDF = %.4f", 
                                doubleExpFit->GetChisquare() / doubleExpFit->GetNDF()));
            stats->AddText(Form("N_{#mu} = %.1f #pm %.1f", 
                                doubleExpFit->GetParameter(0), doubleExpFit->GetParError(0)));
            stats->AddText(Form("N_{acc} = %.1f #pm %.1f", 
                                doubleExpFit->GetParameter(2), doubleExpFit->GetParError(2)));
            stats->AddText(Form("#tau_{acc} fixed = %.4f #mus", tau_accidental));
            stats->Draw();
        }
        
        // Print fit results
        double N_mu = doubleExpFit->GetParameter(0);
        double N_mu_err = doubleExpFit->GetParError(0);
        double tau_mu = doubleExpFit->GetParameter(1);
        double tau_mu_err = doubleExpFit->GetParError(1);
        double N_acc = doubleExpFit->GetParameter(2);
        double N_acc_err = doubleExpFit->GetParError(2);
        double chi2 = doubleExpFit->GetChisquare();
        int ndf = doubleExpFit->GetNDF();
        double chi2_ndf = ndf > 0 ? chi2 / ndf : 0;

        cout << "Double Exponential Fit Results (Michel dt, " << FIT_MIN << "-" << FIT_MAX << " µs):\n";
        cout << "Fit Status: " << fitStatus << " (0 = success)\n";
        cout << Form("τ_μ = %.4f ± %.4f µs", tau_mu, tau_mu_err) << endl;
        cout << Form("N_μ = %.1f ± %.1f", N_mu, N_mu_err) << endl;
        cout << Form("N_acc = %.1f ± %.1f", N_acc, N_acc_err) << endl;
        cout << Form("τ_acc (fixed) = %.4f µs", tau_accidental) << endl;
        cout << Form("χ²/NDF = %.4f", chi2_ndf) << endl;
        cout << "----------------------------------------" << endl;
    } else {
        cout << "Warning: h_dt_michel has insufficient entries (" << h_dt_michel->GetEntries() 
             << "), skipping double exponential fit" << endl;
    }
    
    c->Update();
    plotName = OUTPUT_DIR + "/Michel_dt.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    // Compare different exponential fit start times for double exponential
    if (h_dt_michel->GetEntries() > 5 && doubleExpFit != nullptr) {
        // Define fit start times (1.0 to 4.0 μs in 0.5 μs steps)
        std::vector<double> fit_starts = {1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0};
        std::vector<double> taus, tau_errs, chi2ndfs;
        int best_index = -1;
        double min_chi2ndf = 1e9;
        
        // Perform fits for each start time
        for (int i = 0; i < fit_starts.size(); i++) {
            double fit_start = fit_starts[i];
            double fit_end = 16.0;
            
            TF1* doubleExpFit_var = new TF1(Form("doubleExpFit_var_%.1f", fit_start), ConstrainedDoubleExpFit, fit_start, fit_end, 3);
            doubleExpFit_var->SetParNames("N_{#mu}", "#tau_{#mu}", "N_{acc}");
            
            // Use parameters from main fit as initial values
            double N0_init = doubleExpFit->GetParameter(0);
            double tau_mu_init = doubleExpFit->GetParameter(1);
            double N_acc_init = doubleExpFit->GetParameter(2);
            
            doubleExpFit_var->SetParameters(N0_init, tau_mu_init, N_acc_init);
            doubleExpFit_var->SetParLimits(0, 0, N0_init * 10);
            doubleExpFit_var->SetParLimits(1, 0.1, 20.0);
            doubleExpFit_var->SetParLimits(2, 0, N_acc_init * 10);
            doubleExpFit_var->SetLineColor(i+1);
            
            // Perform fit (quietly)
            int fitStatus = h_dt_michel->Fit(doubleExpFit_var, "QRN+", "", fit_start, fit_end);
            
            // Record results
            double tau = doubleExpFit_var->GetParameter(1);
            double tau_err = doubleExpFit_var->GetParError(1);
            double chi2 = doubleExpFit_var->GetChisquare();
            int ndf = doubleExpFit_var->GetNDF();
            double chi2ndf = (ndf > 0) ? chi2 / ndf : 999;
            
            taus.push_back(tau);
            tau_errs.push_back(tau_err);
            chi2ndfs.push_back(chi2ndf);
            
            if (chi2ndf < min_chi2ndf && fitStatus == 0) {
                min_chi2ndf = chi2ndf;
                best_index = i;
            }
            
            // Print fit results for this range
            cout << Form("Double Exp Fit Range %.1f–%.1f µs:\n", fit_start, fit_end);
            cout << "Fit Status: " << fitStatus << " (0 = success)\n";
            cout << Form("τ_μ = %.4f ± %.4f µs", tau, tau_err) << endl;
            cout << Form("χ²/NDF = %.4f", chi2ndf) << endl;
            cout << "----------------------------------------" << endl;
            
            delete doubleExpFit_var;
        }
        
        // Print best fit result
        if (best_index >= 0) {
            cout << Form("Best Double Exp Fit Range: %.1f–16.0 µs\n", fit_starts[best_index]);
            cout << Form("τ_μ = %.4f ± %.4f µs", taus[best_index], tau_errs[best_index]) << endl;
            cout << Form("χ²/NDF = %.4f (minimum)", chi2ndfs[best_index]) << endl;
            cout << "----------------------------------------" << endl;
        }
        
        // Create comparison plot
        TCanvas* c_comp = new TCanvas("c_comp", "Double Exp Fit Start Time Comparison", 1200, 800);
        c_comp->SetGrid();
        
        // Create pad for the main plot
        TPad* pad = new TPad("pad", "pad", 0, 0, 1, 1);
        pad->Draw();
        pad->cd();
        
        // Create graphs
        TGraph* g_chi2 = new TGraph(fit_starts.size(), &fit_starts[0], &chi2ndfs[0]);
        TGraph* g_tau = new TGraph(fit_starts.size(), &fit_starts[0], &taus[0]);
        
        // Configure chi2 graph (left axis)
        g_chi2->SetTitle("Double Exponential Fit Start Time Comparison");
        g_chi2->GetXaxis()->SetTitle("Fit Start Time (#mus)");
        g_chi2->GetYaxis()->SetTitle("#chi^{2}/ndf");
        g_chi2->SetMarkerStyle(20);
        g_chi2->SetMarkerColor(kBlue);
        g_chi2->SetLineColor(kBlue);
        g_chi2->SetLineWidth(2);
        
        // Configure tau graph (right axis)
        g_tau->SetMarkerStyle(20); // Use filled circle to match chi2 style
        g_tau->SetMarkerColor(kRed);
        g_tau->SetLineColor(kRed);
        g_tau->SetLineWidth(2);
        
        // Draw chi2 first to establish the frame
        g_chi2->Draw("ALP");
        
        // Create right axis
        pad->Update();
        double ymin = pad->GetUymin();
        double ymax = pad->GetUymax();
        
        // Scale tau values to match chi2 plot range
        double tau_min = *min_element(taus.begin(), taus.end()) * 0.9;
        double tau_max = *max_element(taus.begin(), taus.end()) * 1.1;
        double scale = (ymax - ymin)/(tau_max - tau_min);
        double offset = ymin - tau_min * scale;
        
        // Scale the tau graph
        for (int i = 0; i < g_tau->GetN(); i++) {
            double x, y;
            g_tau->GetPoint(i, x, y);
            g_tau->SetPoint(i, x, y * scale + offset);
        }
        
        // Draw tau graph on same pad (line with markers, no error bars)
        g_tau->Draw("LP same");
        
        // Create right axis
        TGaxis* axis = new TGaxis(gPad->GetUxmax(), gPad->GetUymin(),
                                 gPad->GetUxmax(), gPad->GetUymax(),
                                 tau_min, tau_max, 510, "+L");
        axis->SetLineColor(kRed);
        axis->SetLabelColor(kRed);
        axis->SetTitle("#tau_{#mu} (#mus)");
        axis->SetTitleColor(kRed);
        axis->Draw();
        
        // Add legend
        TLegend* leg = new TLegend(0.7, 0.7, 0.9, 0.9);
        leg->AddEntry(g_chi2, "#chi^{2}/ndf", "lp");
        leg->AddEntry(g_tau, "#tau_{#mu}", "lp");
        leg->Draw();
        
        // Save plot
        string compPlotName = OUTPUT_DIR + "/DoubleExpFitStartComparison.png";
        c_comp->SaveAs(compPlotName.c_str());
        cout << "Saved comparison plot: " << compPlotName << endl;
        
        // Clean up
        delete g_chi2;
        delete g_tau;
        delete leg;
        delete axis;
        delete pad;
        delete c_comp;
    }
    else {
        cout << "Skipping double exponential fit start comparison - insufficient entries in dt histogram" << endl;
    }

    // Energy vs dt
    c->Clear();
    h_energy_vs_dt->SetStats(0);
    h_energy_vs_dt->GetXaxis()->SetTitle("dt (#mus)");
    h_energy_vs_dt->Draw("COLZ");
    c->Update();
    plotName = OUTPUT_DIR + "/Michel_Energy_vs_dt.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    // Side Veto Muon
    c->Clear();
    h_side_vp_muon->SetLineColor(kMagenta);
    h_side_vp_muon->Draw();
    c->Update();
    plotName = OUTPUT_DIR + "/Side_Veto_Muon.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    // Top Veto Muon
    c->Clear();
    h_top_vp_muon->SetLineColor(kCyan);
    h_top_vp_muon->Draw();
    c->Update();
    plotName = OUTPUT_DIR + "/Top_Veto_Muon.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    // Trigger Bits Distribution
    c->Clear();
    h_trigger_bits->SetLineColor(kGreen);
    h_trigger_bits->Draw();
    c->Update();
    plotName = OUTPUT_DIR + "/TriggerBits_Distribution.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    // Clean up
    delete h_muon_energy;
    delete h_michel_energy;
    delete h_dt_michel;
    delete h_energy_vs_dt;
    delete h_side_vp_muon;
    delete h_top_vp_muon;
    delete h_trigger_bits;
    delete h_dt_accidental;
    delete h_accidental_energy;
    if (doubleExpFit) delete doubleExpFit;
    delete c;

    cout << "Analysis complete. Results saved in " << OUTPUT_DIR << "/ (*.png)" << endl;
    return 0;
}
