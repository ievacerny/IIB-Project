using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Assertions;

public class CursorBlink : MonoBehaviour {

    public int blinking_rate = 25;
    private int counter = 1;

    private Renderer rend;

	void Start () {
        rend = GetComponent<Renderer>();
        Assert.IsNotNull(rend);
	}
	
	void Update () {
        if (counter % blinking_rate == 0)
        {
            rend.enabled = !rend.enabled;
            counter = 1;
        }
        else
        {
            counter++;
        }
	}
}
