using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;
using System.Diagnostics;
using System.Text;
using System.IO;
using System.Runtime.CompilerServices;
using TMPro;

public class VACommunication : MonoBehaviour
{
	public string relativeFilePath = "response.mp3";
	public string fullPath;
	[SerializeField] private AudioSource source;
	[SerializeField] private TextMeshProUGUI text;
	private Process pythonProcess;
	string pythonInterpreterPath = @"C:\Users\abbas\AppData\Local\Programs\Python\Python39\python.exe";
	string pythonScriptPath = System.IO.Path.Combine(Application.dataPath, "Communication.py");
	private bool playingAudio = false;
	private bool enteredIf = false;

	// Start is called before the first frame update
	void Start()
	{
		fullPath = System.IO.Path.Combine(Application.dataPath, relativeFilePath.Replace("Assets/", ""));
		pythonProcess = new Process
		{
			StartInfo = new ProcessStartInfo
			{
				FileName = pythonInterpreterPath,
				Arguments = $"{pythonScriptPath}",
				UseShellExecute = false,
				RedirectStandardOutput = true,
				RedirectStandardError = true,
				CreateNoWindow = true
			},
			EnableRaisingEvents = true // Allow capturing process exit event
		};

		// Event handler for capturing the process exit event
		pythonProcess.Exited += (sender, e) =>
		{
			string output = pythonProcess.StandardOutput.ReadToEnd();
			string errorOutput = pythonProcess.StandardError.ReadToEnd();

			if (!string.IsNullOrEmpty(output))
			{
				UnityEngine.Debug.Log("Python script output: " + output);
			}

			if (!string.IsNullOrEmpty(errorOutput))
			{
				UnityEngine.Debug.LogError("Python script error: " + errorOutput);
			}

			pythonProcess.Dispose(); // Dispose of the process to free resources
		};
	}

	// Update is called once per frame
	void Update()
	{
		if(Input.GetKeyDown(KeyCode.Q))
		{
			UnityEngine.Debug.Log("Q Pressed");
			StartPythonScript();
			pythonProcess.Start();
			File.WriteAllText("Assets/Captions.txt", "~");
		}

		if (Input.GetKeyDown(KeyCode.W))
		{
			UnityEngine.Debug.Log("W Pressed");
			pythonProcess.Kill();
			pythonProcess.Dispose();
			File.WriteAllText("Assets/Captions.txt", "~");
		}

		text.text = File.ReadAllText("Assets/Captions.txt");
		if(text.text.Contains("|") && !source.isPlaying && !playingAudio && !enteredIf)
		{
			enteredIf = true;
			StartCoroutine(LoadAndProcessAudio(fullPath));
		}
		else if(text.text.Contains("|") && playingAudio && !source.isPlaying)
		{
			enteredIf = false;
			playingAudio = false;
			File.WriteAllText("Assets/Captions.txt", "~");
		}
	}

	private IEnumerator LoadAudio(string path)
    {
        // Use UnityWebRequest to load the audio file
        using (UnityWebRequest www = UnityWebRequestMultimedia.GetAudioClip("file://" + path, AudioType.MPEG))
        {
            yield return www.SendWebRequest();

            if (www.result == UnityWebRequest.Result.ConnectionError || www.result == UnityWebRequest.Result.ProtocolError)
            {
                UnityEngine.Debug.LogError(www.error);
            }
            else
            {
                // Get the AudioClip from the request and assign it to the AudioSource
                AudioClip audioClip = DownloadHandlerAudioClip.GetContent(www);
                source.clip = audioClip;
				source.Play();
            }
        }
    }

	private IEnumerator LoadAndProcessAudio(string path)
    {
		// Start the LoadAudio coroutine and wait for it to finish
        yield return StartCoroutine(LoadAudio(fullPath));

        // This code will execute after LoadAudio has finished
        UnityEngine.Debug.Log("Audio loading complete. Now executing additional code.");

		playingAudio = true;
    }

	void StartPythonScript()
	{
		string pythonInterpreterPath = @"C:\Users\abbas\AppData\Local\Programs\Python\Python39\python.exe";
		string pythonScriptPath = System.IO.Path.Combine(Application.dataPath, "Communication.py");

		pythonProcess = new Process
		{
			StartInfo = new ProcessStartInfo
			{
				FileName = pythonInterpreterPath,
				Arguments = $"{pythonScriptPath}",
				UseShellExecute = false,
				RedirectStandardOutput = true,
				RedirectStandardError = true,
				CreateNoWindow = true
			},
			EnableRaisingEvents = true // Allow capturing process exit event
		};

		// Event handler for capturing the process exit event
		pythonProcess.Exited += (sender, e) =>
		{
			string output = pythonProcess.StandardOutput.ReadToEnd();
			string errorOutput = pythonProcess.StandardError.ReadToEnd();

			if (!string.IsNullOrEmpty(output))
			{
				UnityEngine.Debug.Log("Python script output: " + output);
			}

			if (!string.IsNullOrEmpty(errorOutput))
			{
				UnityEngine.Debug.LogError("Python script error: " + errorOutput);
			}

			pythonProcess.Dispose(); // Dispose of the process to free resources
		};
	}

	// Optionally, you may want to stop the Python process when the Unity application quits
	private void OnApplicationQuit()
	{
		if (pythonProcess != null && !pythonProcess.HasExited)
		{
			pythonProcess.Kill();
			pythonProcess.Dispose();
		}
	}
}
