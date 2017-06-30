#### A little note: this is an ongoing work. I am lazy, and I like to let the world think I am a very busy person.

## Preface - What is this about?

My PhD is focused on a *complex* problem. It is called **person re-identification** (**re-id** for friends).

Re-id consists in the retrieval of the same individual which moves through an environment covered by a video surveillance system.

### Interesting. Now, why is this complicated?

Usually, video surveillance systems are composed by a set of heterogeneous cameras. This implies that:

- the same individual can be captured with different view angles, and under varying lightning conditions;
- two cameras may have different resolution and pre-processing algorithms (i.e. white balance, anti-aliasing, etc.);
- occlusions are just behind the corner.

## And why use ConvNets?

Usually, re-id algorithms use *hand-crafted* features. It means that an expert studies the problem, and extracts a set of features which he consider *meaningful*. From these features, a *signature* of each image is extracted; these signatures are compared, and a result is shown.

Well, this happens in theory.

ConvNets have achieved a lot of interesting results in image recognition. Architectures like GoogLeNet or ResNet have moved the state-of-the-art in image processing towards human-level performances. Therefore, why don't we use ConvNets?

## There are a lot of tools. Why Keras?

Because it is beautiful.
