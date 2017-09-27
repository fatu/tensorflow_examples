package com.olol.tensorflow;

import org.tensorflow.*;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

public class HelloTF {
    public static void main(String[] args) throws Exception {
        try (Graph g = new Graph(); Session s = new Session(g)) {
            String modelDir = HelloTF.class.getClassLoader().getResource("SNN_294_100_sigmoid_softmax.pb").getPath();
            final byte[] graphDef = Files.readAllBytes(Paths.get(modelDir));
            g.importGraphDef(graphDef);
            System.out.println(g.operation("dense_5_input_2").output(0));
            System.out.println(g.operation("output_node0").output(0));
            System.out.println(g.operation("strided_slice_1").output(0));

            long[] shape = {1, 294};
            List<String> lines = Files.readAllLines(Paths.get(HelloTF.class.getClassLoader().getResource("data/test_data_s01_s_jan_dec_2015_298F_edited_head").getPath()));
            long start = System.currentTimeMillis();
            for (String line : lines.subList(1, lines.size())) {
                String[] values = line.split("\\|");
                float[] floats = new float[294];
                for (int i = 0; i < values.length - 2; i++) {
                    floats[i] = Float.parseFloat(values[i + 2]);
                }
                FloatBuffer buffer = getFloatBuffer(floats);
                Tensor inputs = Tensor.create(shape, buffer);
                Tensor output = s.runner().feed("dense_5_input_2", inputs).fetch("output_node0").run().get(0);
                float[] val = output.copyTo(new float[2]);
                System.out.println(val[0] + " " + val[1]);
            }
            System.out.println((System.currentTimeMillis() - start)/(double)lines.size());
        }
    }

    private static FloatBuffer getFloatBuffer(float[] floats) {
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(floats.length * Float.BYTES);
        byteBuffer.order(ByteOrder.nativeOrder());
        FloatBuffer buffer = byteBuffer.asFloatBuffer();
        buffer.put(floats);
        buffer.position(0);
        return buffer;
    }
}
