using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RayCastLineRenderer : MonoBehaviour
{
	[SerializeField] private GameObject agent1;
	[SerializeField] private GameObject agent2;
	LineRenderer lineRenderer;
	// Start is called before the first frame update
	void Start()
	{
		lineRenderer = gameObject.GetComponent<LineRenderer>();
	}

	// Update is called once per frame
	void Update()
	{

	}

	void FixedUpdate()
	{
		// Bit shift the index of the layer (8) to get a bit mask
		int layerMask = 1 << 8;

		// This would cast rays only against colliders in layer 8.
		// But instead we want to collide against everything except layer 8. The ~ operator does this, it inverts a bitmask.
		layerMask = ~layerMask;

		RaycastHit hit;
		lineRenderer.SetPosition(0, transform.position);
		// Does the ray intersect any objects excluding the player layer
		if(Physics.Raycast(transform.position, transform.TransformDirection(Vector3.forward), out hit, Mathf.Infinity, layerMask))
		{
			if(hit.collider.tag == "Agent 1")
			{
				Debug.Log("Agent 1");
				agent1.GetComponent<Outline>().enabled = true;
			}
			else if(hit.collider.tag == "Agent 2")
			{
				Debug.Log("Agent 2");
				agent2.GetComponent<Outline>().enabled = true;
			}
			else
			{
				agent1.GetComponent<Outline>().enabled = false;
				agent2.GetComponent<Outline>().enabled = false;
			}
			lineRenderer.SetPosition(1, transform.forward * hit.distance);
		}
		else
		{
			agent1.GetComponent<Outline>().enabled = false;
			agent2.GetComponent<Outline>().enabled = false;
			lineRenderer.SetPosition(1, transform.forward * 100);
		}
	}
}