package org.mnist.service;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;

public class ImageVisualizationService {

    private final int scaleFactor = 10;

    public ImageVisualizationService() {
    }

    public void visualize(double[] pixelValues, int imageSize) {
        int scaledSize = scaleFactor * imageSize;
        BufferedImage image = new BufferedImage(scaledSize, scaledSize, BufferedImage.TYPE_INT_RGB);

        for (int y = 0; y < imageSize; y++) {
            for (int x = 0; x < imageSize; x++) {
                int pixelIndex = y * imageSize + x;
                int pixelValue = (int) pixelValues[pixelIndex];

                pixelValue = Math.max(0, Math.min(255, pixelValue));

                Color color = new Color(pixelValue, pixelValue, pixelValue);
                for (int i = 0; i < scaleFactor; i++) {
                    for (int j = 0; j < scaleFactor; j++) {
                        image.setRGB(x * scaleFactor + i, y * scaleFactor + j, color.getRGB());
                    }
                }
            }
        }

        SwingUtilities.invokeLater(() -> {
            JFrame frame = new JFrame();
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

            JLabel imageLabel = new JLabel(new ImageIcon(image));
            imageLabel.setPreferredSize(new Dimension(scaledSize, scaledSize));

            frame.getContentPane().add(imageLabel);
            frame.pack();
            frame.setVisible(true);
        });
    }
}
